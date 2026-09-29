import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine_google import Engine, EngineError, InputFile, OutputFile, ProgressEvent  # noqa: E402
import engine_google.metadata as meta  # noqa: E402


class TestDatatypes:
    def test_inputfile_defaults(self):
        f = InputFile(path=Path("test.md"), prompt="hello")
        assert f.path == Path("test.md")
        assert f.prompt == "hello"
        assert f.reference_urls == []
        assert f.metadata == {}

    def test_outputfile_defaults(self):
        o = OutputFile(source_path=Path("test.md"))
        assert o.source_path == Path("test.md")
        assert o.path is None
        assert o.status == "ok"
        assert o.error_msg == ""
        assert o.media_type == ""
        assert o.metadata == {}

    def test_outputfile_error_status(self):
        o = OutputFile(
            source_path=Path("test.md"),
            status="error",
            error_msg="timeout",
            media_type="text",
        )
        assert o.status == "error"
        assert o.error_msg == "timeout"
        assert o.media_type == "text"

    def test_progress_event_defaults(self):
        e = ProgressEvent(message="processing")
        assert e.message == "processing"
        assert e.level == "info"

    def test_engine_error_is_exception(self):
        with pytest.raises(EngineError):
            raise EngineError("test")


class TestMetadata:
    def test_provider_name(self):
        assert meta.PROVIDER_NAME == "Google AI"

    def test_platform(self):
        assert meta.PLATFORM == "google"

    def test_api_key_env_var(self):
        assert meta.API_KEY_ENV_VAR == "GOOGLE_API_KEY"

    def test_api_key_pattern(self):
        assert meta.API_KEY_PATTERN == r"^AIza[A-Za-z0-9_-]{30,}$"

    def test_provider_homepage(self):
        assert meta.PROVIDER_HOMEPAGE == "https://ai.google.dev"

    def test_metadata_matches_engine_class(self):
        assert Engine.PLATFORM == meta.PLATFORM
        assert Engine.PROVIDER_NAME == meta.PROVIDER_NAME
        assert Engine.API_KEY_ENV_VAR == meta.API_KEY_ENV_VAR
        assert Engine.API_KEY_PATTERN == meta.API_KEY_PATTERN


class TestEngineInitPurity:
    def test_init_stores_attributes(self):
        profile = {"endpoint": "gemini-2.0-flash", "media_type": "text"}
        engine = Engine(profile, "/tmp/out")
        assert engine._profile == profile
        assert engine._output_dir == Path("/tmp/out")
        assert engine._api_key is None
        assert engine._on_progress is None
        assert engine._prefix == ""
        assert engine._suffix == ""

    def test_init_extracts_prefix_suffix(self):
        profile = {
            "endpoint": "gemini-2.0-flash",
            "prompt_prefix": "Summarize: ",
            "prompt_suffix": " Keep it brief.",
        }
        engine = Engine(profile, "/tmp/out")
        assert engine._prefix == "Summarize: "
        assert engine._suffix == " Keep it brief."


class TestEnginePreflight:
    def test_missing_endpoint_raises(self):
        engine = Engine({"media_type": "text"}, "/tmp/out")
        with pytest.raises(EngineError, match="Missing 'endpoint'"):
            engine.run([])

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises(self):
        mock_genai = MagicMock()
        with patch.dict("sys.modules", {"google": mock_genai, "google.genai": MagicMock()}):
            engine = Engine({"endpoint": "gemini-2.0-flash"}, "/tmp/out")
            with pytest.raises(EngineError, match="not set"):
                engine.run([])

    def test_genai_not_installed_raises(self):
        with patch.dict("sys.modules", {"google": None}):
            engine = Engine({"endpoint": "gemini-2.0-flash"}, "/tmp/out")
            with pytest.raises(EngineError, match="google-genai SDK not installed"):
                engine.run([])


class TestEngineRun:
    def test_empty_inputs(self, tmp_path):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test_key"}):
            mock_genai = MagicMock()
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            with patch.dict("sys.modules", {"google": mock_genai, "google.genai": MagicMock()}):
                engine = Engine(
                    {"endpoint": "gemini-2.0-flash", "media_type": "text"},
                    tmp_path,
                )
                results = engine.run([])
                assert results == []

    def test_run_calls_progress_callback(self, tmp_path):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test_key"}):
            mock_genai = MagicMock()
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Generated text"
            mock_client.models.generate_content.return_value = mock_response
            mock_genai.Client.return_value = mock_client

            progress_calls = []
            with patch.dict("sys.modules", {"google": mock_genai, "google.genai": MagicMock()}):
                engine = Engine(
                    {"endpoint": "gemini-2.0-flash", "media_type": "text"},
                    tmp_path,
                    on_progress=progress_calls.append,
                )
                engine.run([InputFile(path=Path("b.md"), prompt="test")])

            assert len(progress_calls) >= 1
            assert "Processing Markdown file" in progress_calls[0]

    def test_applies_prefix_suffix(self, tmp_path):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test_key"}):
            mock_genai = MagicMock()
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Result"
            mock_client.models.generate_content.return_value = mock_response
            mock_genai.Client.return_value = mock_client
            with patch.dict("sys.modules", {"google": mock_genai, "google.genai": MagicMock()}):
                engine = Engine(
                    {
                        "endpoint": "gemini-2.0-flash",
                        "media_type": "text",
                        "prompt_prefix": "PREFIX: ",
                        "prompt_suffix": " :SUFFIX",
                    },
                    tmp_path,
                )
                engine.run([InputFile(path=Path("b.md"), prompt="hello")])
                call_args = mock_client.models.generate_content.call_args
                contents = call_args[1]["contents"]
                first_text = contents[0].text if hasattr(contents[0], "text") else str(contents[0])
                assert "PREFIX: hello :SUFFIX" in first_text

    def test_empty_prompt_after_prefix_suffix_is_error(self, tmp_path):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test_key"}):
            mock_genai = MagicMock()
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            with patch.dict("sys.modules", {"google": mock_genai, "google.genai": MagicMock()}):
                engine = Engine(
                    {
                        "endpoint": "gemini-2.0-flash",
                        "media_type": "text",
                        "prompt_prefix": "",
                        "prompt_suffix": "",
                    },
                    tmp_path,
                )
                results = engine.run([InputFile(path=Path("b.md"), prompt="  ")])
                assert len(results) == 1
                assert results[0].status == "error"
                assert "Empty prompt" in results[0].error_msg

    def test_per_markdown_file_error_returns_error_outputfile(self, tmp_path):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test_key"}):
            mock_genai = MagicMock()
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = RuntimeError(
                "API timeout"
            )
            mock_genai.Client.return_value = mock_client
            with patch.dict("sys.modules", {"google": mock_genai, "google.genai": MagicMock()}):
                engine = Engine(
                    {"endpoint": "gemini-2.0-flash", "media_type": "text"},
                    tmp_path,
                )
                results = engine.run([InputFile(path=Path("b.md"), prompt="test")])
                assert len(results) == 1
                assert results[0].status == "error"
                assert "API timeout" in results[0].error_msg

    def test_partial_success_mixed_batch(self, tmp_path):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test_key"}):
            mock_genai = MagicMock()
            mock_client = MagicMock()

            responses = []
            for i in range(3):
                if i == 1:
                    responses.append(RuntimeError("fail"))
                else:
                    r = MagicMock()
                    r.text = "ok"
                    responses.append(r)
            mock_client.models.generate_content.side_effect = responses
            mock_genai.Client.return_value = mock_client
            with patch.dict("sys.modules", {"google": mock_genai, "google.genai": MagicMock()}):
                engine = Engine(
                    {"endpoint": "gemini-2.0-flash", "media_type": "text"},
                    tmp_path,
                )
                markdown_files = [
                    InputFile(path=Path(f"b{i}.md"), prompt="test")
                    for i in range(3)
                ]
                results = engine.run(markdown_files)
                statuses = [r.status for r in results]
                assert statuses.count("ok") == 2
                assert statuses.count("error") == 1

    def test_text_output_saved_as_txt(self, tmp_path):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test_key"}):
            mock_genai = MagicMock()
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Generated text output"
            mock_client.models.generate_content.return_value = mock_response
            mock_genai.Client.return_value = mock_client
            with patch.dict("sys.modules", {"google": mock_genai, "google.genai": MagicMock()}):
                engine = Engine(
                    {"endpoint": "gemini-2.0-flash", "media_type": "text"},
                    tmp_path,
                )
                results = engine.run(
                    [InputFile(path=Path("b.md"), prompt="test")]
                )
                assert len(results) == 1
                assert results[0].status == "ok"
                assert results[0].path is not None
                assert results[0].path.suffix == ".txt"
                assert results[0].path.read_text() == "Generated text output"

    def test_none_text_returns_error(self, tmp_path):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test_key"}):
            mock_genai = MagicMock()
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = None
            mock_client.models.generate_content.return_value = mock_response
            mock_genai.Client.return_value = mock_client
            with patch.dict("sys.modules", {"google": mock_genai, "google.genai": MagicMock()}):
                engine = Engine(
                    {"endpoint": "gemini-2.0-flash", "media_type": "text"},
                    tmp_path,
                )
                results = engine.run(
                    [InputFile(path=Path("b.md"), prompt="test")]
                )
                assert len(results) == 1
                assert results[0].status == "error"
                assert "No content" in results[0].error_msg

    def test_vision_media_type_includes_image_uris(self, tmp_path):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test_key"}):
            mock_genai = MagicMock()
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Vision result"
            mock_client.models.generate_content.return_value = mock_response
            mock_genai.Client.return_value = mock_client
            mock_types = MagicMock()
            mock_part = MagicMock()
            mock_part.text = "text_part"
            mock_types.Part.from_text.return_value = mock_part
            mock_types.Part.from_uri.return_value = MagicMock()

            with patch.dict("sys.modules", {
                "google": mock_genai,
                "google.genai": MagicMock(),
                "google.genai.types": mock_types,
            }):
                engine = Engine(
                    {"endpoint": "gemini-2.0-flash-vision", "media_type": "vision"},
                    tmp_path,
                )
                engine.run([
                    InputFile(
                        path=Path("b.md"),
                        prompt="Describe this",
                        reference_urls=["https://img.example.com/photo.jpg"],
                    )
                ])
                mock_types.Part.from_uri.assert_called_once()


class TestImports:
    def test_init_exports_all_names(self):
        from engine_google import (
            Engine,
            EngineError,
            InputFile,
            OutputFile,
            ProgressEvent,
        )
        assert Engine is not None
        assert InputFile is not None
        assert OutputFile is not None
        assert ProgressEvent is not None
        assert EngineError is not None
