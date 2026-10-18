import os
from datetime import datetime
from pathlib import Path

from .datatypes import EngineError, InputFile, OutputFile, ProgressEvent


class Engine:
    PLATFORM: str = "google"
    PROVIDER_NAME: str = "Google AI"
    PROVIDER_HOMEPAGE: str = "https://ai.google.dev"
    API_KEY_ENV_VAR: str = "GOOGLE_API_KEY"
    API_KEY_PATTERN: str = r"^AIza[A-Za-z0-9_-]{30,}$"

    def __init__(
        self,
        profile: dict,
        output_dir: str | Path,
        api_key: str | None = None,
        on_progress: "Callable[[str], None] | None" = None,
    ):
        self._profile = profile
        self._output_dir = Path(output_dir)
        self._api_key = api_key
        self._on_progress = on_progress
        self._prefix = profile.get("prompt_prefix", "")
        self._suffix = profile.get("prompt_suffix", "")
        self._concurrency = profile.get("concurrency", 1)

    def run(self, inputs: list[InputFile]) -> list[OutputFile]:
        self._validate_preflight()

        from google import genai

        client = genai.Client(api_key=self._resolve_api_key())
        endpoint = self._profile["endpoint"]
        params = dict(self._profile.get("parameters", {}))
        media_type = self._profile.get("media_type", "")
        self._output_dir.mkdir(parents=True, exist_ok=True)

        if self._concurrency > 1:
            from . import runner

            return runner.run_batch(self, inputs)

        results: list[OutputFile] = []
        total = len(inputs)

        for idx, item in enumerate(inputs):
            self._emit(f"Processing Markdown file {idx + 1}/{total}...")
            prompt = f"{self._prefix}{item.prompt}{self._suffix}".strip()
            if not prompt:
                output = OutputFile(
                    source_path=item.path,
                    status="error",
                    error_msg="Empty prompt after applying prefix/suffix",
                    media_type=media_type,
                )
                results.append(output)
                continue

            try:
                contents = self._build_contents(prompt, item, media_type)
                response = client.models.generate_content(
                    model=endpoint,
                    contents=contents,
                )

                if not response.text:
                    output = OutputFile(
                        source_path=item.path,
                        status="error",
                        error_msg="No content returned from Google AI",
                        media_type=media_type,
                    )
                else:
                    ts = self._now_string()
                    dest = self._output_dir / f"{ts}-{item.path.stem}-{idx}.txt"
                    dest.write_text(response.text, encoding="utf-8")
                    output = OutputFile(
                        source_path=item.path,
                        path=dest,
                        status="ok",
                        media_type=media_type,
                    )

            except Exception as exc:
                output = OutputFile(
                    source_path=item.path,
                    status="error",
                    error_msg=str(exc),
                    media_type=media_type,
                )

            results.append(output)

        return results

    def _validate_preflight(self):
        if not self._profile.get("endpoint"):
            raise EngineError("Missing 'endpoint' in profile")
        if (
            isinstance(self._concurrency, bool)
            or not isinstance(self._concurrency, int)
            or self._concurrency < 1
        ):
            raise EngineError(
                f"Invalid 'concurrency' in profile: {self._concurrency!r} — "
                "must be a whole number >= 1"
            )
        try:
            from google import genai  # noqa: F401
        except ImportError:
            raise EngineError(
                "google-genai SDK not installed. Run: pip install google-genai"
            ) from None
        if not self._resolve_api_key():
            raise EngineError(
                f"{self.API_KEY_ENV_VAR} not set in environment or .env file"
            )

    def _resolve_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        return os.environ.get(self.API_KEY_ENV_VAR, "")

    def _now_string(self) -> str:
        return datetime.now().strftime("%y%m%d_%H%M%S")

    def _build_contents(
        self, prompt: str, item: InputFile, media_type: str
    ) -> list:
        from google.genai import types

        parts: list = [types.Part.from_text(text=prompt)]

        if media_type == "vision" and item.reference_urls:
            for url in item.reference_urls:
                parts.append(
                    types.Part.from_uri(
                        file_uri=url,
                        mime_type="image/jpeg",
                    )
                )

        return parts

    def _emit(self, message: str, level: str = "info"):
        if self._on_progress:
            self._on_progress(message)
