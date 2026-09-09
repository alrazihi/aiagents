import json

from gemini_agent_toolkit.observability import Metrics, StructuredLogger

# -------- StructuredLogger (existing) -------- #

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


# -------- StructuredLogger: file output (new) -------- #

def test_file_logging_writes_json(tmp_path):
    log_file = tmp_path / "test.log"
    logger = StructuredLogger(name="test_file", log_path=str(log_file))
    logger.info("event_a", key="value_a")
    logger.error("event_b", detail="failed")
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        payload = json.loads(line)
        assert "timestamp" in payload
        assert "level" in payload
        assert "event" in payload


def test_file_logging_includes_timestamp(tmp_path):
    log_file = tmp_path / "test.log"
    logger = StructuredLogger(name="test_ts", log_path=str(log_file))
    logger.info("my_event", data="123")
    payload = json.loads(log_file.read_text().strip())
    assert payload["timestamp"]
    assert payload["level"] == "INFO"
    assert payload["event"] == "my_event"
    assert payload["data"] == "123"


def test_multiple_loggers_same_name_no_duplicate_handlers():
    """Two StructuredLogger instances with the same name must not add duplicate handlers."""
    logger1 = StructuredLogger(name="dedup_test")
    handler_count_1 = len(logger1.logger.handlers)
    StructuredLogger(name="dedup_test")
    handler_count_2 = len(logger1.logger.handlers)
    assert handler_count_1 == handler_count_2


# -------- Metrics (new) -------- #

def test_metrics_initial_values():
    m = Metrics()
    assert m.tokens_used == 0
    assert m.latency_ms == 0.0
    assert m.tool_calls == 0
    assert m.errors == 0


def test_metrics_record_tool_call():
    m = Metrics()
    m.record_tool_call()
    m.record_tool_call()
    assert m.tool_calls == 2


def test_metrics_record_error():
    m = Metrics()
    m.record_error()
    m.record_error()
    assert m.errors == 2


def test_metrics_to_dict():
    m = Metrics()
    m.record_tool_call()
    m.record_error()
    m.latency_ms = 123.45
    m.tokens_used = 100
    d = m.to_dict()
    assert d == {
        "tokens_used": 100,
        "latency_ms": 123.45,
        "tool_calls": 1,
        "errors": 1,
    }


def test_metrics_record_latency():
    m = Metrics()
    m.latency_ms = 42.0
    assert m.to_dict()["latency_ms"] == 42.0
