import json
from unittest.mock import MagicMock, patch

import google.genai.errors as gerrors
import pytest
from google.genai.errors import (
    ClientError,
    FunctionInvocationError,
    ServerError,
)

from gemini_agent_toolkit.agent import (
    RETRYABLE_STATUSES,
    Agent,
    AgentError,
    _is_non_retryable_error,
    _is_retryable_error,
)
from gemini_agent_toolkit.observability import RateLimiter

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


# -------- Test helpers for creating API errors -------- #

def _retryable_error():
    return ServerError(503, {"error": {"status": "UNAVAILABLE"}})


def _rate_limit_error():
    return ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED"}})


def _permission_error():
    return ClientError(403, {"error": {"status": "PERMISSION_DENIED"}})


def _invalid_arg_error():
    return ClientError(400, {"error": {"status": "INVALID_ARGUMENT"}})


# -------- Fixtures -------- #

@pytest.fixture
def mock_genai():
    with patch("gemini_agent_toolkit.agent.genai") as mock:
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_client.chats.create.return_value = mock_chat
        mock.Client.return_value = mock_client
        yield mock, mock_chat


@pytest.fixture
def mock_sleep():
    with patch("gemini_agent_toolkit.agent.time.sleep") as mock:
        yield mock


@pytest.fixture
def agent(mock_genai, mock_sleep):
    _, mock_chat = mock_genai
    a = Agent(api_key="test-api-key-12345")
    return a, mock_chat


# -------- Error classification tests -------- #

def test_retryable_error_classification():
    assert _is_retryable_error(_retryable_error())
    assert _is_retryable_error(_rate_limit_error())
    assert not _is_retryable_error(_permission_error())
    assert not _is_retryable_error(_invalid_arg_error())


def test_non_retryable_error_classification():
    assert not _is_non_retryable_error(_retryable_error())
    assert not _is_non_retryable_error(_rate_limit_error())
    assert _is_non_retryable_error(_permission_error())
    assert _is_non_retryable_error(_invalid_arg_error())
    assert _is_non_retryable_error(FunctionInvocationError("test"))


def test_retryable_statuses_contains_expected():
    assert "RESOURCE_EXHAUSTED" in RETRYABLE_STATUSES
    assert "UNAVAILABLE" in RETRYABLE_STATUSES


# -------- safe_send tests -------- #

def test_safe_send_returns_on_first_success(mock_genai, mock_sleep):
    _, mock_chat = mock_genai
    response = make_text_response("Hello")
    mock_chat.send_message.return_value = response
    agent = Agent(api_key="test-api-key-12345")
    result = agent.safe_send("Hello")
    assert result is response
    mock_chat.send_message.assert_called_once()


def test_safe_send_retries_on_rate_limit(mock_genai, mock_sleep):
    _, mock_chat = mock_genai
    response = make_text_response("Recovered")
    mock_chat.send_message.side_effect = [
        _rate_limit_error(),
        response,
    ]
    agent = Agent(api_key="test-api-key-12345")
    result = agent.safe_send("Hello")
    assert result is response
    assert mock_chat.send_message.call_count == 2
    mock_sleep.assert_called_once()


def test_safe_send_retries_on_server_error(mock_genai, mock_sleep):
    _, mock_chat = mock_genai
    response = make_text_response("Recovered")
    mock_chat.send_message.side_effect = [
        _retryable_error(),
        response,
    ]
    agent = Agent(api_key="test-api-key-12345")
    result = agent.safe_send("Hello")
    assert result is response
    assert mock_chat.send_message.call_count == 2


def test_safe_send_no_retry_on_permission_denied(mock_genai, mock_sleep):
    _, mock_chat = mock_genai
    mock_chat.send_message.side_effect = _permission_error()
    agent = Agent(api_key="test-api-key-12345")
    with pytest.raises(ClientError):
        agent.safe_send("Hello")
    assert mock_chat.send_message.call_count == 1
    mock_sleep.assert_not_called()


def test_safe_send_no_retry_on_invalid_argument(mock_genai, mock_sleep):
    _, mock_chat = mock_genai
    mock_chat.send_message.side_effect = _invalid_arg_error()
    agent = Agent(api_key="test-api-key-12345")
    with pytest.raises(ClientError):
        agent.safe_send("Hello")
    assert mock_chat.send_message.call_count == 1
    mock_sleep.assert_not_called()


def test_safe_send_no_retry_on_function_invocation_error(mock_genai, mock_sleep):
    _, mock_chat = mock_genai
    mock_chat.send_message.side_effect = FunctionInvocationError("bad function")
    agent = Agent(api_key="test-api-key-12345")
    with pytest.raises(FunctionInvocationError):
        agent.safe_send("Hello")
    assert mock_chat.send_message.call_count == 1
    mock_sleep.assert_not_called()


def test_safe_send_max_retries_exhausted(mock_genai, mock_sleep):
    _, mock_chat = mock_genai
    exc = _rate_limit_error()
    mock_chat.send_message.side_effect = exc
    agent = Agent(api_key="test-api-key-12345")
    with patch("gemini_agent_toolkit.agent.settings") as mock_settings:
        mock_settings.max_retries = 3
        mock_settings.initial_backoff_seconds = 1
        mock_settings.max_backoff_seconds = 10
        with pytest.raises(ClientError):
            agent.safe_send("Hello")
    assert mock_chat.send_message.call_count == 3
    assert mock_sleep.call_count == 2


def test_safe_send_backoff_increases(mock_genai, mock_sleep):
    _, mock_chat = mock_genai
    exc = _rate_limit_error()
    mock_chat.send_message.side_effect = [exc, exc, make_text_response("ok")]
    agent = Agent(api_key="test-api-key-12345")
    with patch("gemini_agent_toolkit.agent.settings") as mock_settings:
        mock_settings.max_retries = 5
        mock_settings.initial_backoff_seconds = 2
        mock_settings.max_backoff_seconds = 100
        agent.safe_send("Hello")
    waits = [call[0][0] for call in mock_sleep.call_args_list]
    assert waits[1] > waits[0]


def test_agent_requires_valid_api_key():
    with pytest.raises(ValueError, match="api_key"):
        Agent(api_key="")
    with pytest.raises(ValueError, match="api_key"):
        Agent(api_key="short")


def test_agent_accepts_valid_api_key(mock_genai, mock_sleep):
    Agent(api_key="valid-api-key-123456789")


# -------- run_task tests -------- #

def test_run_task_returns_final_text(agent):
    a, mock_chat = agent
    mock_chat.send_message.return_value = make_text_response("Final answer here")
    result = a.run_task("Hello")
    assert result == "Final answer here"


def test_run_task_validates_empty_input(agent):
    a, _ = agent
    with pytest.raises(AgentError) as exc_info:
        a.run_task("")
    assert exc_info.value.code == "VALIDATION_ERROR"


def test_run_task_validates_whitespace_input(agent):
    a, _ = agent
    with pytest.raises(AgentError) as exc_info:
        a.run_task("   ")
    assert exc_info.value.code == "VALIDATION_ERROR"


def test_run_task_validates_long_input(agent):
    a, _ = agent
    with pytest.raises(AgentError) as exc_info:
        a.run_task("x" * 5000)
    assert exc_info.value.code == "VALIDATION_ERROR"


def test_run_task_tool_call_allowed(agent, tmp_path, monkeypatch):
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
    a, mock_chat = agent
    a.allowed_tools = ["read_file"]
    mock_chat.send_message.side_effect = [
        make_tool_call_response("execute_command", {"command": "echo hello"}),
        make_text_response("Done"),
    ]
    a.run_task("Do something")
    assert a.metrics.errors >= 1


def test_run_task_handles_safety_filter_block(agent):
    a, mock_chat = agent
    mock_chat.send_message.return_value = make_safety_blocked_response()
    result = a.run_task("Tell me a joke")
    assert "blocked" in result.lower()


def test_run_task_validates_tool_output(agent, tmp_path, monkeypatch):
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
    a, mock_chat = agent
    mock_chat.send_message.return_value = make_text_response("Done")
    a.run_task("Hello task")
    history = a.memory.get_history()
    assert len(history) >= 2
    roles = [h["role"] for h in history]
    assert "user" in roles
    assert "assistant" in roles


def test_run_task_handles_content_none(agent):
    a, mock_chat = agent
    response = MagicMock()
    response.candidates = [MagicMock()]
    response.candidates[0].finish_reason = 1
    response.candidates[0].content = None
    mock_chat.send_message.return_value = response
    result = a.run_task("Hello")
    assert "Error" in result or "No content" in result


def test_run_task_accumulates_text_from_multiple_parts(agent):
    a, mock_chat = agent
    response = MockResponse(parts=[
        MockPart(text="Part 1. "),
        MockPart(text="Part 2."),
    ])
    mock_chat.send_message.return_value = response
    result = a.run_task("Hello")
    assert result == "Part 1. Part 2."


def test_run_task_records_latency(agent):
    a, mock_chat = agent
    mock_chat.send_message.return_value = make_text_response("Done")
    a.run_task("Hello")
    assert a.metrics.latency_ms >= 0.0


# -------- _execute_tool tests -------- #

def _make_tool_call(name, args=None):
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
    a, _ = agent
    result = a._execute_tool(_make_tool_call("read_file", {"invalid_arg": "val"}))
    assert "Error" in result


# -------- AgentError / health_check -------- #

def test_run_task_raises_agent_error_on_empty_input(agent):
    a, _ = agent
    with pytest.raises(AgentError) as exc_info:
        a.run_task("")
    assert exc_info.value.code == "VALIDATION_ERROR"


def test_run_task_raises_agent_error_on_whitespace_input(agent):
    a, _ = agent
    with pytest.raises(AgentError) as exc_info:
        a.run_task("   ")
    assert exc_info.value.code == "VALIDATION_ERROR"


def test_run_task_raises_agent_error_on_oversized_input(agent):
    a, _ = agent
    with pytest.raises(AgentError) as exc_info:
        a.run_task("x" * 5000)
    assert exc_info.value.code == "VALIDATION_ERROR"
    assert "max_length" in exc_info.value.details


def test_agent_error_to_dict():
    err = AgentError("test message", code="TEST_CODE", detail="info")
    assert err.to_dict() == {
        "code": "TEST_CODE",
        "message": "test message",
        "detail": "info",
    }


@patch("gemini_agent_toolkit.agent.genai")
def test_health_check_valid_key(mock_genai):
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.list.return_value = []
    result = Agent.health_check("valid-api-key-1234567890")
    assert result == {"healthy": True, "detail": "OK"}


@patch("gemini_agent_toolkit.agent.genai")
def test_health_check_invalid_key_format(mock_genai):
    result = Agent.health_check("short")
    assert result["healthy"] is False
    assert "Invalid API key" in result["detail"]


@patch("gemini_agent_toolkit.agent.genai")
def test_health_check_api_error(mock_genai):
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.list.side_effect = gerrors.ServerError(
        500, {"error": {"status": "INTERNAL"}}
    )
    result = Agent.health_check("valid-api-key-1234567890")
    assert result["healthy"] is False
    assert "API error" in result["detail"]


def test_safe_send_blocks_rate_limit(agent):
    a, _ = agent
    a.rate_limiter = RateLimiter(max_requests=0, window_seconds=1.0)
    with pytest.raises(RuntimeError, match="Rate limit exceeded"):
        a.safe_send("test")


def test_run_task_logs_rate_limit_event(agent, caplog):
    a, _ = agent
    a.rate_limiter = RateLimiter(max_requests=0, window_seconds=1.0)
    with pytest.raises(RuntimeError):
        a.run_task("Hello")
    logged = any(
        "rate_limit" in record.getMessage()
        for record in caplog.records
    )
    assert logged


# -------- Input validation -------- #

def test_run_task_raises_on_non_string_input(agent):
    a, _ = agent
    with pytest.raises(AgentError) as exc_info:
        a.run_task(None)
    assert exc_info.value.code == "INVALID_TASK_TYPE"


def test_run_task_raises_on_non_string_input_list(agent):
    a, _ = agent
    with pytest.raises(AgentError) as exc_info:
        a.run_task(["not a string"])
    assert exc_info.value.code == "INVALID_TASK_TYPE"


# -------- max_tool_iterations -------- #

def test_run_task_stops_at_max_tool_iterations(agent, tmp_path, monkeypatch):
    a, mock_chat = agent
    (tmp_path / "f.txt").write_text("x")
    monkeypatch.chdir(tmp_path)
    tool_response = make_tool_call_response("read_file", {"file_path": "f.txt"})
    mock_chat.send_message.side_effect = [tool_response] * 20
    with patch("gemini_agent_toolkit.agent.settings") as mock_settings:
        mock_settings.max_tool_iterations = 3
        mock_settings.max_retries = 5
        mock_settings.initial_backoff_seconds = 0
        mock_settings.max_backoff_seconds = 1
        mock_settings.rate_limit_requests = 1000
        result = a.run_task("Read f.txt repeatedly")
    assert "Maximum tool iterations" in result
    assert mock_chat.send_message.call_count <= 4


def test_run_task_logs_task_length_not_full_text(agent, caplog):
    a, mock_chat = agent
    mock_chat.send_message.return_value = make_text_response("Done")
    with caplog.at_level("INFO"):
        a.run_task("Hello world this is a test")
    for record in caplog.records:
        if "task_started" in record.getMessage():
            payload = json.loads(record.getMessage())
            assert "task_length" in payload
            assert "task" not in payload


def test_safe_send_logs_error_type_not_message(agent, mock_genai, mock_sleep, caplog):
    _, mock_chat = mock_genai
    mock_chat.send_message.side_effect = _permission_error()
    a = Agent(api_key="test-api-key-12345")
    with pytest.raises(ClientError):
        a.safe_send("test")
    error_logs = [r for r in caplog.records if "non_retryable" in r.getMessage()]
    assert error_logs
    payload = json.loads(error_logs[-1].getMessage())
    assert "error_type" in payload
    assert "error" not in payload  # No raw error message leaked
