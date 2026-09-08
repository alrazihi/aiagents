import pytest
from gemini_agent_toolkit.guardrails import Guardrails


def test_validate_input_rejects_empty():
    with pytest.raises(ValueError):
        Guardrails.validate_input("")


def test_validate_input_rejects_long_prompt():
    with pytest.raises(ValueError):
        Guardrails.validate_input("x" * 4001)


def test_validate_output_redacts_blocked_patterns():
    text = "Please run rm -rf / and DROP TABLE users"
    result = Guardrails.validate_output(text)
    assert "rm -rf /" not in result
    assert "DROP TABLE" not in result


def test_check_tool_permission_allows_listed_tool():
    assert Guardrails.check_tool_permission("read_file", ["read_file", "write_file"]) is True


def test_check_tool_permission_blocks_unlisted_tool():
    with pytest.raises(PermissionError):
        Guardrails.check_tool_permission("execute_command", ["read_file"])
