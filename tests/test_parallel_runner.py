"""N-in-flight proofs — mirrored from the gold engine's stub suite (phase_3)."""

import sys
import threading
import time
from datetime import datetime as _dt
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine_google import Engine, EngineError, InputFile  # noqa: E402


class _Frozen(_dt):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 18, 12, 0, 0)


def inputs_stub(n=3):
    return [InputFile(path=Path(f"b{i}.md"), prompt=f"p{i}") for i in range(n)]


def base_profile(**extra):
    return {"endpoint": "test/model", "media_type": "image", **extra}


def run_engine(profile, out_dir, impl, probe_calls=None, progress=None, n=3, frozen=True):
    mock_genai = MagicMock()
    mock_genai.genai = mock_genai
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = impl
    mock_genai.Client.return_value = mock_client
    mock_types = MagicMock()
    mock_types.Part.from_text.side_effect = lambda *, text: text
    mock_types.Part.from_uri.side_effect = lambda **kw: kw.get("file_uri", "")
    mock_genai.types = mock_types
    recorder = progress if progress is not None else []
    patches = [
        patch.dict("os.environ", {"GOOGLE_API_KEY": "AIza_test"}),
        patch.dict("sys.modules", {"google": mock_genai, "google.genai": mock_genai}),
    ]
    if frozen:
        patches.append(patch("engine_google.engine.datetime", _Frozen))
    with patches[0], patches[1]:
        ctx = patches[2] if frozen else _NullCtx()
        with ctx:
            engine = Engine(profile, out_dir, on_progress=recorder.append)
            return engine.run(inputs_stub(n)), recorder, mock_client


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def ok_impl(model, contents, **kwargs):
    time.sleep(0.02)
    return MagicMock(text="answer")


class TestParallelRunner:
    def test_result_set_and_order_identical_to_own_concurrency_one(self, tmp_path):
        seq_results, _, _ = run_engine(base_profile(), tmp_path / "seq", ok_impl)
        par_results, _, _ = run_engine(
            base_profile(concurrency=3), tmp_path / "par", ok_impl
        )
        assert [r.status for r in par_results] == [r.status for r in seq_results]
        assert [r.source_path for r in par_results] == [
            r.source_path for r in seq_results
        ]
        assert [r.path.name for r in par_results] == [r.path.name for r in seq_results]
        assert len(par_results) == 3

    def test_more_than_one_job_in_flight(self, tmp_path):
        overlap = []

        def impl(model, contents, **kwargs):
            overlap.append(1)
            time.sleep(0.05)
            overlap.pop()
            return MagicMock(text="a")

        overlap_state = {"cur": 0, "max": 0, "lock": threading.Lock()}

        def watching(model, contents, **kwargs):
            with overlap_state["lock"]:
                overlap_state["cur"] += 1
                overlap_state["max"] = max(overlap_state["max"], overlap_state["cur"])
            try:
                time.sleep(0.05)
                return MagicMock(text="a")
            finally:
                with overlap_state["lock"]:
                    overlap_state["cur"] -= 1

        run_engine(base_profile(concurrency=3), tmp_path, watching)
        assert overlap_state["max"] >= 2

    def test_never_more_than_n_in_flight(self, tmp_path):
        state = {"cur": 0, "max": 0, "lock": threading.Lock()}

        def watching(model, contents, **kwargs):
            with state["lock"]:
                state["cur"] += 1
                state["max"] = max(state["max"], state["cur"])
            try:
                time.sleep(0.05)
                return MagicMock(text="a")
            finally:
                with state["lock"]:
                    state["cur"] -= 1

        run_engine(base_profile(concurrency=2), tmp_path, watching)
        assert state["max"] <= 2

    def test_failing_item_leaves_in_flight_siblings_intact(self, tmp_path):
        def impl(model, contents, **kwargs):
            time.sleep(0.02)
            if "boom" in str(contents):
                raise RuntimeError("google blew up")
            return MagicMock(text="a")

        files = [
            InputFile(path=Path("b0.md"), prompt="fine"),
            InputFile(path=Path("b1.md"), prompt="boom"),
            InputFile(path=Path("b2.md"), prompt="fine"),
        ]
        mock_genai = MagicMock()
        mock_genai.genai = mock_genai
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = impl
        mock_genai.Client.return_value = mock_client
        mock_types = MagicMock()
        mock_types.Part.from_text.side_effect = lambda *, text: text
        mock_types.Part.from_uri.side_effect = lambda **kw: kw.get("file_uri", "")
        mock_genai.types = mock_types
        progress = []
        with patch.dict(
            "os.environ", {"GOOGLE_API_KEY": "AIza_test"}
        ), patch.dict(
            "sys.modules", {"google": mock_genai, "google.genai": mock_genai}
        ), patch(
            "engine_google.engine.datetime", _Frozen
        ):
            engine = Engine(base_profile(concurrency=3), tmp_path, on_progress=progress.append)
            results = engine.run(files)
        statuses = [r.status for r in results]
        assert "error" in statuses and statuses.count("ok") == 2

    def test_no_extra_caller_callargs_leak(self, tmp_path):
        results, _, mock_client = run_engine(
            base_profile(concurrency=3, parameters={}), tmp_path, ok_impl
        )
        for call in mock_client.models.generate_content.call_args_list:
            assert "concurrency" not in str(call.kwargs)
        assert len(results) == 3

    def test_invalid_concurrency_raises_loudly(self, tmp_path):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "AIza_test"}):
            engine = Engine(base_profile(concurrency=-1), tmp_path)
            with pytest.raises(EngineError, match="concurrency"):
                engine.run([])

    def test_retryable_failures_trip_breaker_and_leave_unstarted_absent(self, tmp_path):
        def limited(model, contents, **kwargs):
            error = RuntimeError("429 Too Many Requests")
            error.status_code = 429
            raise error

        results, progress, client = run_engine(
            base_profile(
                concurrency=2,
                retry_on_429=0,
                circuit_breaker_max=2,
            ),
            tmp_path,
            limited,
            n=6,
        )
        calls = client.models.generate_content.call_count
        assert calls <= 3
        assert len(results) == calls
        assert all(result.status == "error" for result in results)
        assert any("Circuit breaker tripped" in str(event) for event in progress)
