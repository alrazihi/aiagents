# ADR-005: Guarded agent integration

## Status

Implemented

## Context

The original `Agent` class created instances of `Guardrails`,
`ConversationMemory`, and `StructuredLogger` but never actually used
them during `run_task`. Guardrails validation was completely bypassed —
the model could call any tool without permission checks, submit
arbitrarily long prompts, and produce unredacted dangerous output.
Observability events were never logged during task execution. Memory
was instantiated but never fed user/assistant messages.

## Decision

1. `run_task` now calls `Guardrails.validate_input(task)` before sending
   to the model, rejecting empty, whitespace-only, or over-length prompts.

2. `_execute_tool` calls `Guardrails.check_tool_permission(name,
   self.allowed_tools)` for every tool invocation. Disallowed tools are
   logged as errors, return a "blocked" message to the model, and
   increment the error metric.

3. All tool output is passed through `Guardrails.validate_output(result)`
   to redact dangerous patterns before sending back to the model.

4. `run_task` logs structured events: `task_started`, `tool_call`,
   `tool_calls_completed`, `task_completed` — each with a UUID
   correlation ID for traceability.

5. `ConversationMemory` records user and assistant messages via
   `memory.add("user", ...)` and `memory.add("assistant", ...)`.

6. `Metrics` tracks `tool_calls` and `errors` for runtime observability.

## Consequences

- Guardrails are enforced at the use-case layer (`run_task`/`_execute_tool`),
  not just as standalone utilities.
- Every task execution produces a correlation ID visible in logs,
  enabling traceability.
- Tool allow-listing prevents the model from executing arbitrary tools.
- Output redaction prevents the model from seeing or propagating
  dangerous command patterns.

## Evidence

- `gemini_agent_toolkit/agent.py:133-158` — `run_task` with validation, logging, memory
- `gemini_agent_toolkit/agent.py:182-200` — `_execute_tool` with permission check
- `gemini_agent_toolkit/agent.py:205-260` — `_consume_response` with output validation
- `tests/test_agent.py::test_run_task_validates_empty_input`
- `tests/test_agent.py::test_run_task_blocks_disallowed_tool`
- `tests/test_agent.py::test_run_task_logs_start_and_completion`
- `tests/test_agent.py::test_run_task_adds_to_memory`
