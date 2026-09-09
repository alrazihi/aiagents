# ADR-001: Bounded retry with error classification

## Status

Implemented

## Context

The original `Agent.safe_send` method used an unbounded `while True` retry loop
that caught **all** exceptions (`except Exception`) and retried indefinitely
with a 10-second sleep. This meant:

- Authentication failures (HTTP 401/403) retried forever, wasting API calls.
- Invalid argument errors (HTTP 400) retried forever, never succeeding.
- Genuine bugs in user code masked by infinite retries.
- No way to set a maximum retry count from configuration.

## Decision

1. Classify Gemini API errors into **retryable** and **non-retryable** tuples
   at module level (`RETRYABLE_ERRORS`, `NON_RETRYABLE_ERRORS`).
2. Replace the `while True` loop with a bounded `for` loop over
   `max_retries` (default 5, configurable via `settings.max_retries`).
3. Non-retryable errors (`PermissionDenied`, `Unauthenticated`,
   `InvalidArgument`, `NotFound`) are re-raised immediately.
4. Retryable errors (`ResourceExhausted`, `ServiceUnavailable`,
   `InternalServerError`, `DeadlineExceeded`) are retried with exponential
   backoff, respecting the API-provided `retryDelay` when available.

## Consequences

- Prevents infinite retry loops on permanent failures.
- Reduces unnecessary API calls on authentication or authorization errors.
- Makes retry behavior observable through structured logging.
- The `assert last_exc is not None` before `raise` is a mypy requirement;
  the loop is guaranteed to set `last_exc` at least once.

## Evidence

- `gemini_agent_toolkit/agent.py:83-128` — retry loop and error tuples
- `gemini_agent_toolkit/config.py:9` — `max_retries` setting
- `tests/test_agent.py::test_safe_send_max_retries_exhausted`
- `tests/test_agent.py::test_safe_send_no_retry_on_permission_denied`
