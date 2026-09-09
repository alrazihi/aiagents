import json

from gemini_agent_toolkit.observability import Metrics, RateLimiter, StructuredLogger

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


# -------- StructuredLogger: log levels -------- #

def test_structured_logger_respects_log_level(caplog):
    logger = StructuredLogger(name="test_level", log_level="ERROR")
    logger.info("should_not_appear")
    logger.error("should_appear")
    messages = [r.message for r in caplog.records]
    assert any("should_appear" in m for m in messages)
    assert not any("should_not_appear" in m for m in messages)


def test_handler_lock_prevents_duplicates_under_concurrency():
    import threading
    """Creating many loggers concurrently should never produce duplicate handlers."""
    results: list[int] = []

    def make_logger():
        logger = StructuredLogger(name="concurrent_dedup")
        results.append(len(logger.logger.handlers))

    threads = [threading.Thread(target=make_logger) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(h == results[0] for h in results)
    assert results[0] <= 2  # StreamHandler + optional file handler


# -------- RateLimiter -------- #

def test_rate_limiter_allows_under_limit():
    limiter = RateLimiter(max_requests=5, window_seconds=1.0)
    allowed = [limiter.acquire() for _ in range(5)]
    assert all(allowed)


def test_rate_limiter_blocks_over_limit():
    limiter = RateLimiter(max_requests=3, window_seconds=1.0)
    results = [limiter.acquire() for _ in range(5)]
    assert results == [True, True, True, False, False]


def test_rate_limiter_window_expires():
    import time as _time
    limiter = RateLimiter(max_requests=2, window_seconds=0.3)
    assert limiter.acquire()
    assert limiter.acquire()
    assert not limiter.acquire()
    _time.sleep(0.4)
    assert limiter.acquire()


# -------- Metrics -------- #

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


def test_metrics_thread_safe_concurrent_updates():
    import threading
    m = Metrics()
    threads = [
        threading.Thread(
            target=lambda: [m.record_tool_call() for _ in range(100)]
        )
        for _ in range(10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.tool_calls == 1000


def test_metrics_add_tokens_and_latency():
    m = Metrics()
    m.add_tokens(50)
    m.add_tokens(30)
    m.add_latency(10.5)
    m.add_latency(20.0)
    assert m.tokens_used == 80
    assert m.latency_ms == 30.5


def test_metrics_record_error_thread_safe():
    import threading
    m = Metrics()
    threads = [
        threading.Thread(
            target=lambda: [m.record_error() for _ in range(50)]
        )
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.errors == 400
