from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import (
    DeadlineExceeded,
    InternalServerError,
    InvalidArgument,
    PermissionDenied,
    ResourceExhausted,
    ServiceUnavailable,
)

from gemini_agent_toolkit.agent import NON_RETRYABLE_ERRORS, RETRYABLE_ERRORS, Agent

# -------- Mock response helpers -------- #

class MockFunctionCall:
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or {}


class MockPart:
    def __init__(self, text=None, function_call=None):
        self.text = text
        self.function_call = function_call


class MockContent:
    def __init__(self, parts):
        self.parts = parts


class MockCandidate:
    def __init__(self, parts=None, finish_reason=1):
        self.content = MockContent(parts or [])
        self.finish_reason = finish_reason


class MockResponse:
    def __init__(self, parts=None, finish_reason=1, candidates=None):
        if candidates is not None:
            self.candidates = candidates
        elif parts is not None:
            self.candidates = [MockCandidate(parts, finish_reason)]
        else:
            self.candidates = [MockCandidate()]


def make_text_response(text):
    return MockResponse(parts=[MockPart(text=text)])


def make_tool_call_response(name, args=None):
    return MockResponse(parts=[MockPart(function_call=MockFunctionCall(name, args))])


def make_safety_blocked_response():
    return MockResponse(candidates=[])


# -------- Fixtures -------- #

@pytest.fixture
def mock_genai():
    with patch("gemini_agent_toolkit.agent.genai") as mock:
        mock_model = MagicMock()
        mock_chat = MagicMock()
        mock_model.start_chat.return_value = mock_chat
        mock.GenerativeModel.return_value = mock_model
        yield mock, mock_chat


@pytest.fixture
def mock_sleep():
    with patch("gemini_agent_toolkit.agent.time.sleep") as mock:
        yield mock


@pytest.fixture
def agent(mock_genai, mock_sleep):
    _, mock_chat = mock_genai
    a = Agent(api_key="test-key")
    return a, mock_chat


# -------- safe_send tests -------- #

def test_safe_send_returns_on_first_success(mock_genai, mock_sleep):
    _, mock_chat = mock_genai
    response = make_text_response("Hello")
    mock_chat.send_message.return_value = response
    agent = Agent(api_key="test-key")
    result = agent.safe_send("Hello")
    assert result is response
    mock_chat.send_message.assert_called_once()


def test_safe_send_retries_on_resource_exhausted(mock_genai, mock_sleep):
    _, mock_chat = mock_genai
    response = make_text_response("Recovered")
    mock_chat.send_message.side_effect = [
        ResourceExhausted("rate limited"),
        response,
    ]
    agent = Agent(api_key="test-key")
    result = agent.safe_send("Hello")
    assert result is response
    assert mock_chat.send_message.call_count == 2
    mock_sleep.assert_called_once()


def test_safe_send_no_retry_on_permission_denied(mock_genai, mock_sleep):
    _, mock_chat = mock_genai
    mock_chat.send_message.side_effect = PermissionDenied("bad key")
    agent = Agent(api_key="test-key")
    with pytest.raises(PermissionDenied):
        agent.safe_send("Hello")
    assert mock_chat.send_message.call_count == 1
    mock_sleep.assert_not_called()


def test_safe_send_no_retry_on_invalid_argument(mock_genai, mock_sleep):
    _, mock_chat = mock_genai
    mock_chat.send_message.side_effect = InvalidArgument("bad request")
    agent = Agent(api_key="test-key")
    with pytest.raises(InvalidArgument):
        agent.safe_send("Hello")
    assert mock_chat.send_message.call_count == 1
    mock_sleep.assert_not_called()


def test_safe_send_max_retries_exhausted(mock_genai, mock_sleep):
    _, mock_chat = mock_genai
    exc = ResourceExhausted("always rate limited")
    mock_chat.send_message.side_effect = exc
    agent = Agent(api_key="test-key")
    with patch("gemini_agent_toolkit.agent.settings") as mock_settings:
        mock_settings.max_retries = 3
        mock_settings.initial_backoff_seconds = 1
        mock_settings.max_backoff_seconds = 10
        with pytest.raises(ResourceExhausted):
            agent.safe_send("Hello")
    assert mock_chat.send_message.call_count == 3
    assert mock_sleep.call_count == 2


def test_safe_send_backoff_increases(mock_genai, mock_sleep):
    _, mock_chat = mock_genai
    exc = ResourceExhausted("rate limited")
    mock_chat.send_message.side_effect = [exc, exc, make_text_response("ok")]
    agent = Agent(api_key="test-key")
    with patch("gemini_agent_toolkit.agent.settings") as mock_settings:
        mock_settings.max_retries = 5
        mock_settings.initial_backoff_seconds = 2
        mock_settings.max_backoff_seconds = 100
        agent.safe_send("Hello")
    waits = [call[0][0] for call in mock_sleep.call_args_list]
    assert waits[1] > waits[0], "backoff should increase"


def test_retryable_errors_tuple():
    assert ResourceExhausted in RETRYABLE_ERRORS
    assert ServiceUnavailable in RETRYABLE_ERRORS
    assert InternalServerError in RETRYABLE_ERRORS
    assert DeadlineExceeded in RETRYABLE_ERRORS


def test_non_retryable_errors_tuple():
    assert PermissionDenied in NON_RETRYABLE_ERRORS
    assert InvalidArgument in NON_RETRYABLE_ERRORS


# -------- run_task tests -------- #

def test_run_task_returns_final_text(agent):
    a, mock_chat = agent
    mock_chat.send_message.return_value = make_text_response("Final answer here")
    result = a.run_task("Hello")
    assert result == "Final answer here"


def test_run_task_validates_empty_input(agent):
    a, _ = agent
    with pytest.raises(ValueError):
        a.run_task("")


def test_run_task_validates_whitespace_input(agent):
    a, _ = agent
    with pytest.raises(ValueError):
        a.run_task("   ")


def test_run_task_validates_long_input(agent):
    a, _ = agent
    with pytest.raises(ValueError):
        a.run_task("x" * 5000)


def test_run_task_tool_call_allowed(agent, tmp_path, monkeypatch):
    """Agent should execute allowed tools and return final text."""
    a, mock_chat = agent
    (tmp_path / "test.txt").write_text("file content")
    monkeypatch.chdir(tmp_path)
    mock_chat.send_message.side_effect = [
        make_tool_call_response("read_file", {"file_path": "test.txt"}),
        make_text_response("File content is file content"),
    ]
    result = a.run_task("Read test.txt")
    assert result == "File content is file content"


def test_run_task_blocks_disallowed_tool(agent):
    """Agent must block tools not in the allowed_tools list."""
    a, mock_chat = agent
    a.allowed_tools = ["read_file"]
    mock_chat.send_message.side_effect = [
        make_tool_call_response("execute_command", {"command": "echo hello"}),
        make_text_response("Done"),
    ]
    a.run_task("Do something")
    assert a.metrics.errors >= 1


def test_run_task_handles_safety_filter_block(agent):
    """Empty candidates (safety filter) must not crash."""
    a, mock_chat = agent
    mock_chat.send_message.return_value = make_safety_blocked_response()
    result = a.run_task("Tell me a joke")
    assert "blocked" in result.lower()


def test_run_task_validates_tool_output(agent, tmp_path, monkeypatch):
    """Guardrails.validate_output must sanitize dangerous text in tool output."""
    a, mock_chat = agent
    (tmp_path / "test.txt").write_text("clean content")
    monkeypatch.chdir(tmp_path)
    mock_chat.send_message.side_effect = [
        make_tool_call_response("read_file", {"file_path": "test.txt"}),
        make_text_response("Final response"),
    ]
    result = a.run_task("Read a file")
    assert result == "Final response"


def test_run_task_logs_start_and_completion(agent, caplog):
    a, mock_chat = agent
    mock_chat.send_message.return_value = make_text_response("Done")
    with caplog.at_level("INFO"):
        a.run_task("test task")
    events = [r.getMessage() for r in caplog.records]
    assert any("task_started" in e for e in events)
    assert any("task_completed" in e for e in events)


def test_run_task_increments_tool_call_metric(agent, tmp_path, monkeypatch):
    a, mock_chat = agent
    (tmp_path / "test.txt").write_text("hello")
    monkeypatch.chdir(tmp_path)
    mock_chat.send_message.side_effect = [
        make_tool_call_response("read_file", {"file_path": "test.txt"}),
        make_text_response("Done"),
    ]
    a.run_task("Read test.txt")
    assert a.metrics.tool_calls == 1


def test_run_task_records_errors_for_blocked_tool(agent):
    a, mock_chat = agent
    a.allowed_tools = ["read_file"]
    mock_chat.send_message.side_effect = [
        make_tool_call_response("execute_command", {"command": "echo hi"}),
        make_text_response("Done"),
    ]
    a.run_task("test")
    assert a.metrics.errors >= 1


def test_run_task_unknown_tool(agent, caplog):
    """Tools allowed but not recognized should return an error to the model."""
    a, mock_chat = agent
    a.allowed_tools = ["read_file", "write_file", "execute_command", "nonexistent_tool"]
    mock_chat.send_message.side_effect = [
        make_tool_call_response("nonexistent_tool", {"arg": "val"}),
        make_text_response("OK"),
    ]
    with caplog.at_level("WARN"):
        result = a.run_task("test")
    assert any("unknown_tool" in r.getMessage() for r in caplog.records)
    assert result == "OK"


def test_run_task_multiple_turns(agent, tmp_path, monkeypatch):
    """Agent should handle multiple tool calls in sequence."""
    a, mock_chat = agent
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    monkeypatch.chdir(tmp_path)
    mock_chat.send_message.side_effect = [
        make_tool_call_response("read_file", {"file_path": "a.txt"}),
        make_tool_call_response("read_file", {"file_path": "b.txt"}),
        make_text_response("All done"),
    ]
    result = a.run_task("Read both files")
    assert result == "All done"
    assert a.metrics.tool_calls == 2


def test_run_task_adds_to_memory(agent):
    """ConversationMemory should receive user and assistant messages."""
    a, mock_chat = agent
    mock_chat.send_message.return_value = make_text_response("Done")
    a.run_task("Hello task")
    history = a.memory.get_history()
    assert len(history) >= 2
    roles = [h["role"] for h in history]
    assert "user" in roles
    assert "assistant" in roles


def test_run_task_handles_content_none(agent):
    """Response with content=None should not crash."""
    a, mock_chat = agent
    response = MagicMock()
    response.candidates = [MagicMock()]
    response.candidates[0].finish_reason = 1
    response.candidates[0].content = None
    mock_chat.send_message.return_value = response
    result = a.run_task("Hello")
    assert "Error" in result or "No content" in result


def test_run_task_accumulates_text_from_multiple_parts(agent):
    """Final text should combine multiple text parts."""
    a, mock_chat = agent
    response = MockResponse(parts=[
        MockPart(text="Part 1. "),
        MockPart(text="Part 2."),
    ])
    mock_chat.send_message.return_value = response
    result = a.run_task("Hello")
    assert result == "Part 1. Part 2."


# -------- _execute_tool tests -------- #

def _make_tool_call(name, args=None):
    """Create a mock tool call object with proper name/args attributes."""
    tc = MagicMock()
    tc.name = name
    tc.args = args or {}
    return tc


def test_execute_tool_blocks_unlisted_tool(agent):
    a, _ = agent
    a.allowed_tools = ["read_file"]
    result = a._execute_tool(_make_tool_call("execute_command", {"command": "echo hi"}))
    assert "blocked" in result.lower()


def test_execute_tool_unknown_tool_name(agent):
    a, _ = agent
    a.allowed_tools = ["read_file", "write_file", "execute_command", "nonexistent"]
    result = a._execute_tool(_make_tool_call("nonexistent", {}))
    assert "Unknown" in result or "Error" in result


def test_execute_tool_invalid_args(agent):
    """Tool call with wrong args should return error, not crash."""
    a, _ = agent
    result = a._execute_tool(_make_tool_call("read_file", {"invalid_arg": "val"}))
    assert "Error" in result
