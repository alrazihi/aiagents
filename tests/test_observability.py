import json
import pytest
from gemini_agent_toolkit.observability import StructuredLogger


def test_emit_info_event(caplog):
    logger = StructuredLogger(name="test")
    logger._emit("INFO", "task_started", task_id="1")
    assert any("task_started" in record.message for record in caplog.records)


def test_emit_error_event(caplog):
    logger = StructuredLogger(name="test")
    logger._emit("ERROR", "task_failed", task_id="1")
    assert any("task_failed" in record.message for record in caplog.records)


def test_payload_is_valid_json(caplog):
    logger = StructuredLogger(name="test")
    logger._emit("INFO", "event", key="value")
    record = next(r for r in caplog.records if "event" in r.message)
    payload = json.loads(record.message)
    assert payload["event"] == "event"
    assert payload["key"] == "value"
