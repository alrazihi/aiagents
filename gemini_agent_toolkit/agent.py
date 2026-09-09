"""Agent that orchestrates a Gemini LLM with tool-calling, guardrails,
conversation memory, and structured observability.

Uses the ``google.genai`` SDK (v2.x).
"""

from __future__ import annotations

import time
import uuid

import google.genai as genai
import google.genai.errors as gerrors
import google.genai.types as types

from gemini_agent_toolkit import tools
from gemini_agent_toolkit.config import settings
from gemini_agent_toolkit.guardrails import Guardrails
from gemini_agent_toolkit.memory import ConversationMemory
from gemini_agent_toolkit.observability import Metrics, RateLimiter, StructuredLogger

RETRYABLE_STATUSES = frozenset({
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "DEADLINE_EXCEEDED",
    "INTERNAL",
})

NON_RETRYABLE_STATUSES = frozenset({
    "PERMISSION_DENIED",
    "UNAUTHENTICATED",
    "INVALID_ARGUMENT",
    "NOT_FOUND",
})


class AgentError(Exception):
    """Base exception with a structured error code for programmatic handling."""

    def __init__(self, message: str, code: str = "AGENT_ERROR", **details: object) -> None:
        super().__init__(message)
        self.code = code
        self.details = details

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": str(self),
            **self.details,
        }


def _is_retryable_error(exc: Exception) -> bool:
    """Return True if *exc* represents a transient API failure that should
    be retried with backoff.
    """
    if isinstance(exc, gerrors.ServerError):
        return True
    if isinstance(exc, gerrors.APIError):
        status = getattr(exc, "status", None)
        code = getattr(exc, "code", None)
        if status in RETRYABLE_STATUSES or code == 429:
            return True
    return False


def _is_non_retryable_error(exc: Exception) -> bool:
    """Return True if *exc* represents a permanent failure that should not
    be retried (e.g. bad credentials, invalid request).
    """
    if isinstance(exc, (
        gerrors.FunctionInvocationError,
        gerrors.UnknownApiResponseError,
        gerrors.UnknownFunctionCallArgumentError,
        gerrors.UnsupportedFunctionError,
    )):
        return True
    if isinstance(exc, gerrors.ClientError):
        status = getattr(exc, "status", None)
        code = getattr(exc, "code", None)
        if status in RETRYABLE_STATUSES or code == 429:
            return False
        return True
    return False


def _build_tool_declarations() -> list[types.Tool]:
    """Build explicit function declarations for the Gemini API.

    The new ``google.genai`` SDK does not auto-convert Python functions;
    schemas must be declared explicitly.  Each parameter type is restricted
    to ``STRING`` (the only type our tools accept).
    """
    def _str_schema(desc: str) -> types.Schema:
        return types.Schema(
            type=types.Type.STRING,
            description=desc,
        )

    return [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="read_file",
                description="Read a file from the current working directory.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"file_path": _str_schema("Path to the file to read.")},
                    required=["file_path"],
                ),
            ),
            types.FunctionDeclaration(
                name="write_file",
                description="Write content to a file in the current working directory.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "file_path": _str_schema("Path to the file to write."),
                        "content": _str_schema("Content to write to the file."),
                    },
                    required=["file_path", "content"],
                ),
            ),
            types.FunctionDeclaration(
                name="execute_command",
                description="Execute a shell command in the current working directory.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "command": _str_schema("The command to execute."),
                    },
                    required=["command"],
                ),
            ),
        ])
    ]


class Agent:
    """Agent that orchestrates a Gemini LLM with tool-calling, guardrails,
    conversation memory, and structured observability.

    Parameters
    ----------
    api_key:
        Google Gemini API key.
    guardrails:
        Guardrails instance (or class) providing input/output/tool validation.
        Defaults to a new ``Guardrails()``.
    memory:
        ConversationMemory instance for tracking turns.  Defaults to a new
        instance with ``settings.max_turns``.
    allowed_tools:
        List of tool names the agent may invoke.  Tools called by the model
        that are not in this list raise ``PermissionError`` and are reported
        back to the model as an error.
    """

    def __init__(
        self,
        api_key: str,
        guardrails: Guardrails | None = None,
        memory: ConversationMemory | None = None,
        allowed_tools: list[str] | None = None,
    ):
        if not api_key or len(api_key) < 10:
            raise ValueError("api_key must be a non-empty string of at least 10 characters")

        self._client = genai.Client(api_key=api_key)

        self.chat = self._client.chats.create(
            model=settings.gemini_model,
            config=types.GenerateContentConfig(
                tools=_build_tool_declarations(),
                max_output_tokens=settings.max_response_length,
            ),
        )

        self.logger = StructuredLogger(name="agent", log_level=settings.log_level)
        self.metrics = Metrics()
        self.memory = memory or ConversationMemory(max_turns=settings.max_turns)
        self.guardrails = guardrails or Guardrails()
        self.allowed_tools = allowed_tools or [
            "read_file", "write_file", "execute_command",
        ]
        self.rate_limiter = RateLimiter(
            max_requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )

    @classmethod
    def health_check(cls, api_key: str) -> dict[str, object]:
        """Quick self-test: verify the API key is valid and the client connects.

        Returns a dict with ``healthy`` (bool) and ``detail`` (str).
        """
        if not api_key or len(api_key) < 10:
            return {"healthy": False, "detail": "Invalid API key format"}
        try:
            client = genai.Client(api_key=api_key)
            client.models.list()
            return {"healthy": True, "detail": "OK"}
        except gerrors.APIError as e:
            return {"healthy": False, "detail": f"API error: {e.status}"}
        except Exception as e:
            return {"healthy": False, "detail": f"Unexpected error: {type(e).__name__}"}

    # ======================================================
    # SAFE MESSAGE SENDER WITH CONFIGURABLE RETRY + BACKOFF
    # ======================================================
    def safe_send(self, payload, max_retries: int | None = None):
        """Send a message to the Gemini API with bounded retry/backoff.

        Retries only on transient (retryable) API errors.  Non-retryable
        errors such as authentication failures or invalid arguments are
        re-raised immediately.  After ``max_retries`` attempts the last
        retryable exception is re-raised.
        """
        if max_retries is None:
            max_retries = settings.max_retries

        if not self.rate_limiter.acquire():
            self.logger.error(
                "rate_limit_exceeded",
                max_requests=settings.rate_limit_requests,
                window_seconds=settings.rate_limit_window_seconds,
            )
            raise RuntimeError(
                f"Rate limit exceeded: {settings.rate_limit_requests} requests "
                f"per {settings.rate_limit_window_seconds}s"
            )

        backoff = settings.initial_backoff_seconds
        max_backoff = settings.max_backoff_seconds
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                return self.chat.send_message(payload)

            except Exception as exc:
                if _is_non_retryable_error(exc) or not _is_retryable_error(exc):
                    self.logger.error(
                        "api_non_retryable_error",
                        attempt=attempt,
                        max_retries=max_retries,
                        error=str(exc)[:500],
                    )
                    raise

                last_exc = exc

                retry_delay = None
                details = getattr(exc, "details", None)
                if isinstance(details, dict):
                    err_details = details.get("error", {}).get("details", [])
                    if err_details and isinstance(err_details[0], dict):
                        retry_delay = err_details[0].get("retryDelay", {}).get("seconds")

                wait_time = retry_delay if retry_delay else backoff
                wait_time = min(wait_time, max_backoff)

                self.logger.warn(
                    "api_retry",
                    attempt=attempt,
                    max_retries=max_retries,
                    wait_seconds=wait_time,
                    error=str(exc)[:500],
                )

                if attempt < max_retries:
                    time.sleep(wait_time)
                    backoff = min(int(backoff * 1.7), max_backoff)

        assert last_exc is not None
        raise last_exc

    # ======================================================
    # RUN TASK
    # ======================================================
    def run_task(self, task: str) -> str:
        """Execute *task* via the Gemini model, calling tools as needed.

        Returns the final text response from the model.
        """
        try:
            self.guardrails.validate_input(task)
        except ValueError as e:
            raise AgentError(
                str(e), code="VALIDATION_ERROR", max_length=settings.max_prompt_length
            ) from e

        correlation_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        self.logger.info(
            "task_started",
            task=task,
            correlation_id=correlation_id,
        )

        self.memory.add("user", task)

        response = self.safe_send(task)
        final_text = self._consume_response(response, correlation_id)

        self.memory.add("assistant", final_text)

        latency_ms = (time.perf_counter() - start_time) * 1000
        self.metrics.add_latency(latency_ms)

        self.logger.info(
            "task_completed",
            task=task,
            correlation_id=correlation_id,
            final_text_length=len(final_text),
            tool_calls=self.metrics.tool_calls,
            errors=self.metrics.errors,
            latency_ms=round(latency_ms, 2),
        )
        return final_text

    # ======================================================
    # TOOL EXECUTION
    # ======================================================
    def _execute_tool(self, tool_call) -> str:
        name = tool_call.name
        args = {k: v for k, v in tool_call.args.items()}

        try:
            self.guardrails.check_tool_permission(name, self.allowed_tools)

            if name == "read_file":
                return tools.read_file(**args)
            elif name == "write_file":
                return tools.write_file(**args)
            elif name == "execute_command":
                return tools.execute_command(**args)
            else:
                self.logger.warn("unknown_tool", tool=name)
                return f"Error: Unknown tool: {name}"

        except PermissionError as e:
            self.metrics.record_error()
            self.logger.error(
                "tool_permission_denied",
                tool=name,
                error=str(e),
            )
            return f"Tool blocked: {e}"

        except Exception as e:
            self.metrics.record_error()
            self.logger.error(
                "tool_error",
                tool=name,
                error=str(e),
            )
            return f"Error executing tool '{name}': {e}"

    # ======================================================
    # RESPONSE CONSUMPTION
    # ======================================================
    def _consume_response(self, response, correlation_id: str) -> str:
        """Process a model response, executing tool calls as needed.

        Handles safety-filter blocks (empty candidates) and returns the
        accumulated final text.
        """
        final_text = ""

        while True:
            if not response.candidates:
                self.logger.error(
                    "safety_filter_blocked",
                    correlation_id=correlation_id,
                )
                return "Error: Response was blocked by safety filters."

            candidate = response.candidates[0]

            if candidate.finish_reason == 1:
                pass
            elif candidate.finish_reason == 2:
                self.logger.warn("max_tokens_reached", correlation_id=correlation_id)

            content = candidate.content
            if content is None:
                return "Error: No content in response."

            has_tool_call = False
            tool_responses = []

            for part in content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    has_tool_call = True
                    tool_call = part.function_call
                    self.metrics.record_tool_call()
                    self.logger.info(
                        "tool_call",
                        tool=tool_call.name,
                        correlation_id=correlation_id,
                    )

                    result = self._execute_tool(tool_call)
                    safe_result = self.guardrails.validate_output(result)

                    tool_responses.append({
                        "function_response": {
                            "name": tool_call.name,
                            "response": {"output": safe_result},
                        }
                    })

            if has_tool_call:
                response = self.safe_send(tool_responses)
                continue

            for part in content.parts:
                if hasattr(part, "text") and part.text:
                    final_text += part.text

            self.logger.info(
                "tool_calls_completed",
                tool_calls=self.metrics.tool_calls,
                correlation_id=correlation_id,
            )
            return final_text
