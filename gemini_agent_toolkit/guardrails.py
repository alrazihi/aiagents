"""Input/output guardrails for safer agent behavior."""

from __future__ import annotations

import re

MAX_PROMPT_LENGTH = 4000

BLOCKED_PATTERNS = [
    re.compile(r"rm\s+-rf\s+/", re.IGNORECASE),
    re.compile(r"rm\s+-rf\s+\.\*", re.IGNORECASE),
    re.compile(r"DROP\s+TABLE", re.IGNORECASE),
    re.compile(r"DELETE\s+FROM\s+\w+\s+WHERE\s+", re.IGNORECASE),
    re.compile(r"DROP\s+DOLUMN", re.IGNORECASE),
    re.compile(r"TRUNCATE\s+TABLE", re.IGNORECASE),
    re.compile(r">\s*/dev/sd", re.IGNORECASE),
    re.compile(r"format\s+[a-z]:", re.IGNORECASE),
    re.compile(r"chmod\s+-r", re.IGNORECASE),
]

DEFAULT_ALLOWED_TOOLS = ["read_file", "write_file", "execute_command"]


class Guardrails:
    """Guardrails for validating agent inputs, outputs, and tool access.

    All methods are classmethods so they can be used without instantiation,
    making them easy to inject or mock in tests.
    """

    BLOCKED_PATTERNS = BLOCKED_PATTERNS

    @classmethod
    def validate_input(cls, prompt: str) -> bool:
        """Validate that a prompt is non-empty and within length limits.

        Raises ValueError if the prompt is empty, whitespace-only, or exceeds
        ``MAX_PROMPT_LENGTH`` characters.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty")
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt exceeds maximum allowed length of {MAX_PROMPT_LENGTH} characters"
            )
        return True

    @classmethod
    def validate_output(cls, text: str) -> str:
        """Redact known-dangerous patterns from output text."""
        sanitized = text or ""
        for pattern in cls.BLOCKED_PATTERNS:
            if pattern.search(sanitized):
                sanitized = pattern.sub("[REDACTED]", sanitized)
        return sanitized

    @classmethod
    def check_tool_permission(
        cls, tool_name: str, allowed_tools: list[str] | None = None
    ) -> bool:
        """Check whether *tool_name* is in the allow-list.

        Raises PermissionError if the tool is not allowed.
        """
        allowed = allowed_tools if allowed_tools is not None else DEFAULT_ALLOWED_TOOLS
        if tool_name not in allowed:
            raise PermissionError(f"Tool not allowed: {tool_name}")
        return True
