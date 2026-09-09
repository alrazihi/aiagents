"""Agent memory system with conversation history and long-term storage."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ConversationMemory:
    """Thread-safe conversation history with configurable turn limits.

    A ``threading.Lock`` serialises ``add``, ``get_history``, and ``clear``
    so that concurrent callers (e.g. multiple async tasks sharing an agent)
    cannot observe a half-trimmed list or lose messages to a race between
    append and trim.
    """

    def __init__(self, max_turns: int = 20):
        self.messages: list[Message] = []
        self.max_turns = max_turns
        self._lock = threading.Lock()

    def add(self, role: str, content: str, metadata: dict | None = None) -> None:
        with self._lock:
            self.messages.append(Message(role=role, content=content, metadata=metadata or {}))
            if len(self.messages) > self.max_turns * 2:
                self.messages = self.messages[-self.max_turns * 2 :]

    def get_history(self) -> list[dict]:
        with self._lock:
            return [m.to_dict() for m in self.messages]

    def clear(self) -> None:
        with self._lock:
            self.messages = []


class LongTermMemory:
    """File-backed key-value store with atomic writes.

    ``save`` writes to a temporary file in the same directory and then
    calls ``os.replace`` to atomically swap it into place.  A
    ``threading.Lock`` serialises writes so that concurrent saves to the
    same key (e.g. from multiple async tasks) do not produce
    ``PermissionError`` on platforms where ``os.replace`` is not
    concurrent-safe (notably Windows).
    """

    def __init__(self, storage_path: str = ".memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def save(self, key: str, value: dict) -> None:
        with self._lock:
            target = self.storage_path / f"{key}.json"
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.storage_path), suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(value, f, indent=2)
                os.replace(tmp_path, str(target))
            except BaseException:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

    def load(self, key: str) -> dict | None:
        target = self.storage_path / f"{key}.json"
        if not target.exists():
            return None
        data: dict = json.loads(target.read_text())
        return data
