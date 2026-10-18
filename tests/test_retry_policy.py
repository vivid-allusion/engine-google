"""Stub tests for retry limits and provider Retry-After handling."""

import sys
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine_google import Engine
from engine_google.retry_policy import request_with_retry


def test_numeric_and_http_date_retry_after_are_honored(tmp_path):
    http_date = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30))
    for value, lower, upper in (("0.5", 0.5, None), (http_date, 20, 31)):
        engine = Engine({"retry_on_429": 1, "retry_base_delay": 0}, tmp_path)
        response = SimpleNamespace(
            status_code=429, headers={"Retry-After": value}
        )
        error = RuntimeError("429 Too Many Requests")
        error.response = response
        operation = MagicMock(side_effect=[error, "ok"])
        with patch("engine_google.retry_policy.time.sleep") as sleep, patch(
            "engine_google.retry_policy.random.uniform", return_value=0
        ):
            assert request_with_retry(engine, operation, "[1/1] input.md") == "ok"
        assert operation.call_count == 2
        assert sleep.call_count == 1
        delay = sleep.call_args.args[0]
        assert delay >= lower
        if upper is not None:
            assert delay <= upper
