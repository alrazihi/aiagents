"""Observability utilities: structured logging and metrics."""

from __future__ import annotations

import json
import logging
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

    def __init__(self, name: str = "agent", log_path: str | None = None):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
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
        self.tokens_used: int = 0
        self.latency_ms: float = 0.0
        self.tool_calls: int = 0
        self.errors: int = 0

    def record_tool_call(self) -> None:
        self.tool_calls += 1

    def record_error(self) -> None:
        self.errors += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "tool_calls": self.tool_calls,
            "errors": self.errors,
        }
