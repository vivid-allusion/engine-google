"""N-in-flight execution — mirrors the gold engine's runner.

`run_batch` caps jobs in flight structurally: at most the profile's
`concurrency` futures are submitted at any moment. The per-item body is
this engine's own sequential body, verbatim (build contents, generate,
save the text output); results are written by input index; one failing
item never disturbs jobs still in flight. This engine has no
breaker/retry keys to share — the stop flag exists for the same
mirror-shape contract.
"""

import concurrent.futures
import threading

from .datatypes import OutputFile
from .retry_policy import classify_error, request_with_retry


class BatchState:
    """Thread-safe stop flag and rate-limit circuit for one parallel batch."""

    def __init__(self, circuit_max: int):
        self.circuit_max = circuit_max
        self.consecutive_conflicts = 0
        self._lock = threading.Lock()
        self._stop = False

    def is_stopped(self) -> bool:
        with self._lock:
            return self._stop

    def halt(self):
        with self._lock:
            self._stop = True

    def note_success(self):
        with self._lock:
            if not self._stop:
                self.consecutive_conflicts = 0

    def note_rate_limited(self) -> tuple[bool, int]:
        with self._lock:
            self.consecutive_conflicts += 1
            count = self.consecutive_conflicts
            return count >= self.circuit_max, count


def _record_failure(engine, state, exc: Exception, prefix: str):
    kind = classify_error(exc)
    if kind == "fatal":
        state.halt()
        engine._emit(
            f"{prefix} 🛑 Fatal provider error — aborting batch. ({exc})",
            level="error",
        )
    elif kind == "retryable":
        tripped, count = state.note_rate_limited()
        if tripped:
            state.halt()
            engine._emit(
                f"{prefix} 🛑 Circuit breaker tripped after {count} consecutive "
                "rate-limited failures — aborting batch.",
                level="error",
            )
    else:
        state.note_success()


def run_item(engine, client, item, idx, total, state) -> list[OutputFile]:
    media_type = engine._profile.get("media_type", "")
    prefix = f"[{idx + 1}/{total}] {item.path.name}"
    engine._emit(f"{prefix} 📝 Markdown")
    prompt = f"{engine._prefix}{item.prompt}{engine._suffix}".strip()
    if not prompt:
        return [
            OutputFile(
                source_path=item.path,
                status="error",
                error_msg="Empty prompt after applying prefix/suffix",
                media_type=media_type,
            )
        ]
    try:
        contents = engine._build_contents(prompt, item, media_type)
        response = request_with_retry(
            engine,
            lambda: client.models.generate_content(
                model=engine._profile["endpoint"],
                contents=contents,
            ),
            prefix,
        )

        if not response.text:
            state.note_success()
            return [
                OutputFile(
                    source_path=item.path,
                    status="error",
                    error_msg="No content returned from Google AI",
                    media_type=media_type,
            )
            ]
        ts = engine._now_string()
        dest = engine._output_dir / f"{ts}-{item.path.stem}-{idx}.txt"
        dest.write_text(response.text, encoding="utf-8")
        state.note_success()
        return [
            OutputFile(
                source_path=item.path,
                path=dest,
                status="ok",
                media_type=media_type,
            )
        ]
    except Exception as exc:
        _record_failure(engine, state, exc, prefix)
        return [
            OutputFile(
                source_path=item.path,
                status="error",
                error_msg=str(exc),
                media_type=media_type,
            )
        ]


def run_batch(engine, inputs, client) -> list[OutputFile]:
    engine._output_dir.mkdir(parents=True, exist_ok=True)
    state = BatchState(int(engine._profile.get("circuit_breaker_max", 3)))

    results: list[list[OutputFile] | None] = [None] * len(inputs)
    inflight_cap = max(1, engine._concurrency)

    with concurrent.futures.ThreadPoolExecutor(max_workers=inflight_cap) as pool:
        pending: dict[concurrent.futures.Future, int] = {}
        next_idx = 0
        while pending or (next_idx < len(inputs) and not state.is_stopped()):
            while (
                not state.is_stopped()
                and next_idx < len(inputs)
                and len(pending) < inflight_cap
            ):
                future = pool.submit(
                    run_item, engine, client, inputs[next_idx], next_idx, len(inputs), state
                )
                pending[future] = next_idx
                next_idx += 1
            if not pending:
                break
            done, _ = concurrent.futures.wait(pending, return_when="FIRST_COMPLETED")
            for future in done:
                results[pending.pop(future)] = future.result()

    return [out for item in results if item is not None for out in item]
