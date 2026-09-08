"""Input/output guardrails for safer agent behavior."""

from __future__ import annotations

import re
from typing import Optional


class Guardrails:
    BLOCKED_PATTERNS = [
        re.compile(r"rm -rf /"),
        re.compile(r"DROP TABLE"),
        re.compile(r"DELETE FROM .* WHERE .*"),
    ]

    @classmethod
    def validate_input(cls, prompt: str) -> bool:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty")
        if len(prompt) > 4000:
            raise ValueError("Prompt exceeds maximum allowed length")
        return True

    @classmethod
    def validate_output(cls, text: str) -> str:
        sanitized = text or ""
        for pattern in cls.BLOCKED_PATTERNS:
            if pattern.search(sanitized):
                sanitized = pattern.sub("[REDACTED]", sanitized)
        return sanitized

    @classmethod
    def check_tool_permission(cls, tool_name: str, allowed_tools: Optional[list[str]] = None) -> bool:
        if allowed_tools is None:
            allowed_tools = ["read_file", "write_file", "execute_command"]
        if tool_name not in allowed_tools:
            raise PermissionError(f"Tool not allowed: {tool_name}")
        return True
