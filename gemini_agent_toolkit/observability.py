"""Observability utilities: structured logging, metrics, and rate limiting."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class StructuredLogger:
    """JSON-lines structured logger with optional file output.

    Each ``StructuredLogger`` instance resolves a single underlying
    ``logging.Logger`` by name.  Handlers are attached only once per
    logger to prevent duplicate log lines when multiple instances share
    the same name.
    """

    _initialized: set[str] = set()
    _handler_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        name: str = "agent",
        log_path: str | None = None,
        log_level: str = "INFO",
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(level)
        with StructuredLogger._handler_lock:
            if name not in StructuredLogger._initialized:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                self.logger.addHandler(handler)
                StructuredLogger._initialized.add(name)
        self._log_path = Path(log_path) if log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def _emit(self, level: str, event: str, **kwargs: Any) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "event": event,
            **kwargs,
        }
        line = json.dumps(payload, ensure_ascii=False)
        if level == "ERROR":
            self.logger.error(line)
        elif level == "WARN":
            self.logger.warning(line)
        else:
            self.logger.info(line)
        if self._log_path:
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def info(self, event: str, **kwargs: Any) -> None:
        self._emit("INFO", event, **kwargs)

    def warn(self, event: str, **kwargs: Any) -> None:
        self._emit("WARN", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._emit("ERROR", event, **kwargs)


class Metrics:
    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self.tokens_used: int = 0
        self.latency_ms: float = 0.0
        self.tool_calls: int = 0
        self.errors: int = 0

    def record_tool_call(self) -> None:
        with self._lock:
            self.tool_calls += 1

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def add_tokens(self, count: int) -> None:
        with self._lock:
            self.tokens_used += count

    def add_latency(self, ms: float) -> None:
        with self._lock:
            self.latency_ms += ms

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tokens_used": self.tokens_used,
                "latency_ms": self.latency_ms,
                "tool_calls": self.tool_calls,
                "errors": self.errors,
            }


class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Allows at most *max_requests* calls within a rolling *window_seconds*
    window.  ``acquire()`` returns ``True`` if the call is allowed or
    ``False`` if it would exceed the limit.
    """

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()

    def acquire(self) -> bool:
        now = time.monotonic()
        with self._lock:
            cutoff = now - self._window
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            if len(self._timestamps) >= self._max:
                return False
            self._timestamps.append(now)
            return True
