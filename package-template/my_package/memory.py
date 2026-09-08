"""Agent memory system with conversation history and long-term storage."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ConversationMemory:
    def __init__(self, max_turns: int = 20):
        self.messages: List[Message] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str, metadata: Optional[dict] = None) -> None:
        self.messages.append(Message(role=role, content=content, metadata=metadata or {}))
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-self.max_turns * 2 :]

    def get_history(self) -> List[dict]:
        return [m.to_dict() for m in self.messages]

    def clear(self) -> None:
        self.messages = []


class LongTermMemory:
    def __init__(self, storage_path: str = ".memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save(self, key: str, value: dict) -> None:
        target = self.storage_path / f"{key}.json"
        target.write_text(json.dumps(value, indent=2))

    def load(self, key: str) -> Optional[dict]:
        target = self.storage_path / f"{key}.json"
        if not target.exists():
            return None
        return json.loads(target.read_text())
