# ADR-008: Migrate to google.genai SDK

## Status

Implemented

## Context

The `google-generativeai` package is deprecated by Google. Every import
produces a `FutureWarning`:

```
All support for the `google.generativeai` package has ended. It will
no longer be receiving updates or bug fixes. Please switch to the
`google.genai` package as soon as possible.
```

The deprecated SDK's error types (`google.api_core.exceptions.ResourceExhausted`,
`PermissionDenied`, etc.) were used for retry classification. The new SDK
has a different error hierarchy (`google.genai.errors.ServerError`,
`ClientError`, `APIError`).

The new SDK also requires explicit `FunctionDeclaration` schemas for tool
calling — it does not auto-convert Python functions.

## Decision

1. Replace `google-generativeai` with `google-genai` (v2.x) in
   `pyproject.toml` and `dev-requirements.txt`.

2. Replace `genai.configure()` + `genai.GenerativeModel()` +
   `model.start_chat()` with `genai.Client()` + `client.chats.create()`.

3. Replace `google.api_core.exceptions` with `google.genai.errors` for
   error classification. Retryable errors are now determined by
   `ServerError` (5xx) or `ClientError` with `status == "RESOURCE_EXHAUSTED"`
   (429). Non-retryable errors are `ClientError` (other 4xx), 
   `FunctionInvocationError`, and related ValueError subclasses.

4. Add explicit `FunctionDeclaration` schemas for `read_file`,
   `write_file`, and `execute_command` via `_build_tool_declarations()`.
   Each parameter is typed as `STRING` — the only type our tools accept.

5. Add API key validation in `Agent.__init__` (minimum 10 characters).

## Consequences

- Eliminates the deprecation warning.
- Uses the maintained, current SDK.
- Error classification is now based on API `status` strings and HTTP
  codes, which is more robust than relying on a specific set of exception
  classes.
- Tool schemas are explicit and auditable, but more verbose.
- API key validation catches obviously invalid keys at construction time.

## Evidence

- `gemini_agent_toolkit/agent.py` — full migration
- `gemini_agent_toolkit/pyproject.toml` — `google-genai` dependency
- `tests/test_agent.py` — updated fixtures and error types
