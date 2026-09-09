import pytest

from gemini_agent_toolkit.guardrails import (
    DEFAULT_ALLOWED_TOOLS,
    MAX_PROMPT_LENGTH,
    Guardrails,
)

# -------- Input validation (existing + expanded) -------- #

def test_validate_input_rejects_empty():
    with pytest.raises(ValueError):
        Guardrails.validate_input("")


def test_validate_input_rejects_long_prompt():
    with pytest.raises(ValueError):
        Guardrails.validate_input("x" * (MAX_PROMPT_LENGTH + 1))


def test_validate_input_rejects_whitespace_only():
    with pytest.raises(ValueError):
        Guardrails.validate_input("   ")


def test_validate_input_accepts_valid_prompt():
    assert Guardrails.validate_input("Hello, how are you?") is True


def test_validate_input_accepts_max_length():
    assert Guardrails.validate_input("x" * MAX_PROMPT_LENGTH) is True


# -------- Output validation (existing + expanded) -------- #

def test_validate_output_redacts_blocked_patterns():
    text = "Please run rm -rf / and DROP TABLE users"
    result = Guardrails.validate_output(text)
    assert "rm -rf /" not in result
    assert "DROP TABLE" not in result
    assert "[REDACTED]" in result


def test_validate_output_case_insensitive():
    """Guardrails must catch lowercase, uppercase, and mixed case."""
    patterns_to_test = [
        "rm -rf /",
        "RM -RF /",
        "Rm -rF /",
        "DROP TABLE users;",
        "drop table users;",
        "Drop Table Users;",
        "delete from users where id=1",
        "DELETE FROM users WHERE id=1",
    ]
    for pattern in patterns_to_test:
        result = Guardrails.validate_output(f"do something {pattern}")
        assert "[REDACTED]" in result, f"Pattern not redacted: {pattern}"
        assert pattern not in result, f"Pattern leaked: {pattern}"


def test_validate_output_no_redaction_for_safe_text():
    text = "Hello, how are you?"
    result = Guardrails.validate_output(text)
    assert result == text


def test_validate_output_handles_none():
    assert Guardrails.validate_output(None) == ""


def test_validate_output_handles_empty():
    assert Guardrails.validate_output("") == ""


def test_validate_output_redacts_multiple_patterns():
    text = "rm -rf / && DROP TABLE users; delete from logs where 1=1"
    result = Guardrails.validate_output(text)
    assert "rm -rf /" not in result
    assert "DROP TABLE" not in result
    assert "delete from" not in result


# -------- Tool permission checks (existing + expanded) -------- #

def test_check_tool_permission_allows_listed_tool():
    assert Guardrails.check_tool_permission("read_file", ["read_file", "write_file"]) is True


def test_check_tool_permission_blocks_unlisted_tool():
    with pytest.raises(PermissionError):
        Guardrails.check_tool_permission("execute_command", ["read_file"])


def test_check_tool_permission_blocks_unknown_tool():
    with pytest.raises(PermissionError):
        Guardrails.check_tool_permission("dangerous_tool", DEFAULT_ALLOWED_TOOLS)


def test_check_tool_permission_default_allows_core_tools():
    for tool in DEFAULT_ALLOWED_TOOLS:
        assert Guardrails.check_tool_permission(tool) is True


def test_check_tool_permission_empty_allowlist():
    with pytest.raises(PermissionError):
        Guardrails.check_tool_permission("read_file", [])
