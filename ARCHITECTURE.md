# Architecture — Gemini Agent Toolkit

## 1. Problem

Building an agentic workflow with Google Gemini requires composing several
concerns: LLM call management (retry/backoff), tool execution (file I/O and
shell commands), input/output safety, conversation state, evaluation, and
observability. Existing solutions either wrap everything in a monolithic
framework or require assembling disparate libraries by hand.

This toolkit provides **composable, testable building blocks** in a single
Python package, with safety guardrails and observability integrated by
default.

## 2. Goals

- Be a **library**, not a service — importable and composable.
- **Safety first** — path sandboxing, command blocking, input/output
  redaction, and tool allow-listing are enforced by default.
- **Observable by default** — every task produces structured JSON logs
  with a correlation ID and metrics.
- **Testable** — every component has unit and integration tests; no
  real API key required for testing (Gemini is mocked).
- **Configurable** — retry limits, backoff, model, and guardrail
  thresholds are set via `pydantic-settings` (env vars or `.env`).

## 3. Non-goals

- Does not provide a web server, REST API, or HTTP endpoints.
- Does not provide authentication or authorization for multi-tenant
  access (the agent runs locally with a single API key).
- Does not provide a persistent database backend (storage is
  file-based via `LongTermMemory`).
- Does not implement CQRS, event sourcing, or saga patterns.

## 4. Architecture

```
                         ┌─────────────────────────────────┐
                         │           CLI (main.py /          │
                         │          __main__.py)            │
                         └──────────┬──────────┬────────────┘
                                    │          │
                         ┌──────────▼──┐   ┌───▼────────────┐
                         │   Agent     │   │EvaluationHarness│
                         │  (agent.py) │   │  (evaluator.py)  │
                         └──┬──────┬───┴───┬──────────────┘
                            │      │       │
              ┌─────────────▼──┐  │  ┌────▼──────────────┐
              │ Guardrails     │  │  │StructuredLogger   │
              │ (guardrails.py)│  │  │(observability.py)  │
              └─────────────┬──┘  │  └────┬──────────────┘
                            │     │       │
                    ┌───────▼─────▼──┐    │
                    │     Tools      │    │
                    │   (tools.py)   │    │
                    └───────┬─────────┘    │
                            │              │
                    ┌───────▼─────────────▼┐
                    │  Memory               │
                    │ (memory.py)           │
                    └──────────────────────┘
```

### Module layout

| Module            | Responsibility                                      |
|-------------------|----------------------------------------------------|
| `agent.py`        | LLM orchestration: retry/backoff, tool loop,      |
|                   | guardrails integration, observability, memory.     |
| `tools.py`        | File read/write and command execution with         |
|                   | path sandboxing and command blocking.                |
| `guardrails.py`   | Input length validation, output pattern redaction,  |
|                   | tool allow-list enforcement.                          |
| `memory.py`       | Conversation history (thread-safe) and             |
|                   | file-backed long-term storage (atomic writes).     |
| `observability.py`| Structured JSON logging and in-process metrics.     |
| `config.py`       | `pydantic-settings` configuration singleton.       |
| `evaluator.py`    | Task-case evaluation harness for accuracy/latency. |

### Dependency direction

```
agent ──► tools
agent ──► guardrails
agent ──► memory
agent ──► observability
agent ──► config
__main__ ──► agent
evaluator ──► (agent passed in, no direct dependency)
```

No module depends on another at import time. The `Agent` class
**injects** all collaborators via constructor parameters with defaults,
making the architecture effectively dependency-injected.

## 5. Data flow

1. **CLI entry** (`__main__.py`) parses `--directory` and `--task`,
   changes to the directory, loads `.env`, and creates an `Agent`.

2. **`Agent.run_task(task)`** is called:
   a. `Guardrails.validate_input()` rejects empty/oversized prompts.
   b. A UUID `correlation_id` is generated and logged.
   c. The task is sent to the Gemini model via `safe_send()`.
   d. `safe_send` retries retryable API errors with exponential
      backoff (bounded by `max_retries`).
   e. The response is consumed by `_consume_response`:
      - If the model issues tool calls, each is checked against
        `Guardrails.check_tool_permission()`, executed, and the
        output is redacted by `Guardrails.validate_output()`.
      - Tool responses are sent back to the model.
      - The loop continues until the model returns a text response
        or all candidates are exhausted.
   f. The final text is added to `ConversationMemory`.
   g. Completion is logged with metrics.

3. **Observability**: each event is a JSON line with `timestamp`,
   `level`, `event`, `correlation_id`, and domain-specific fields.
   Optional file output can be enabled via `log_path`.

## 6. Security boundaries

| Boundary              | Mechanism                                          |
|-----------------------|---------------------------------------------------|
| **API key**           | Loaded from `.env` (git-ignored). Never logged.   |
| **Tool execution**    | Command allow/deny list; `shell=False`;            |
|                     | metacharacter rejection; 30s timeout.              |
| **File access**       | `realpath` + `commonpath` sandbox; rejects         |
|                     | absolute paths and `..` traversal.                 |
| **Tool allow-list**   | `Guardrails.check_tool_permission()` enforces      |
|                     | the `allowed_tools` set on every tool call.        |
| **Input validation**  | Empty/whitespace/oversized prompts rejected.      |
| **Output redaction**  | Case-insensitive regex blocks `rm -rf`,            |
|                     | `DROP TABLE`, `DELETE FROM`, etc.                  |
| **Error handling**    | Tool errors return user-facing strings, not        |
|                     | stack traces or internal details.                  |
| **Secrets in logs**   | API key is never logged or included in payloads.   |

## 7. Persistence model

### ConversationMemory (in-memory)
- Python `list[Message]` with a `threading.Lock`.
- Bounded by `max_turns * 2` messages (one turn = one user + assistant pair).
- When the limit is exceeded, the oldest messages are trimmed.
- Not persisted to disk — cleared on process exit.

### LongTermMemory (file-backed)
- JSON files stored in a configurable directory (`.memory/` by default).
- One file per key: `{key}.json`.
- Writes are **atomic**: data is written to a temp file, then
  `os.replace()` swaps it into place. This prevents partial-write
  corruption if the process crashes or if two threads race.
- A `threading.Lock` serialises writes on the same `LongTermMemory`
  instance (required on Windows where `os.replace` is not
  concurrent-safe).

## 8. Important architectural decisions

| ADR | Title                                          |
|-----|------------------------------------------------|
| 001 | Bounded retry with error classification        |
| 002 | Shell-free command execution                   |
| 003 | Path sandboxing with realpath and commonpath   |
| 004 | Thread-safe memory with atomic file writes     |
| 005 | Guarded agent integration                      |
| 006 | Case-insensitive guardrail patterns             |
| 007 | pyproject.toml without BOM                     |

## 9. Failure handling

### Transient API errors
`safe_send` retries `ResourceExhausted`, `ServiceUnavailable`,
`InternalServerError`, and `DeadlineExceeded` with exponential backoff
up to `max_retries` (default 5). Non-retryable errors
(`PermissionDenied`, `InvalidArgument`, `NotFound`) raise immediately.

### Tool failures
`_execute_tool` catches all exceptions, logs them as `tool_error` or
`tool_permission_denied`, and returns a user-facing error string to the
model. The error metric is incremented. The agent continues its loop.

### Safety filter blocks
If the Gemini model returns empty candidates (response blocked by
safety filters), `_consume_response` returns a descriptive error
message instead of crashing on `IndexError`.

### Command timeouts
`execute_command` enforces a 30-second timeout via
`subprocess.TimeoutExpired`. The command is killed and an error is
returned.

## 10. Concurrency model

| Component              | Thread-safety                                   |
|------------------------|-------------------------------------------------|
| `ConversationMemory`   | `threading.Lock` on all mutations and reads.   |
| `LongTermMemory`      | `threading.Lock` + atomic `os.replace` writes.  |
| `StructuredLogger`    | Thread-safe (Python `logging` is thread-safe).  |
| `Metrics`             | Not thread-safe (single-agent use case).        |
| `Agent`               | Not thread-safe (single-agent use case).        |

The `Agent` is designed for single-agent, single-thread execution.
`ConversationMemory` and `LongTermMemory` have locks so they can be
shared across threads if needed (e.g., in a web service wrapping the
agent).

## 11. Testing strategy

- **Unit tests**: test each module in isolation.
  `agent.py` tests mock the `google.generativeai` module entirely.
- **Integration tests**: test the agent flow with mocked Gemini
  responses (tool calls → tool execution → final text).
- **Security boundary tests**: path traversal, symlink escape, prefix
  attack, shell injection, command blocking, tool permission denial.
- **Failure path tests**: retry exhaustion, non-retryable errors,
  safety filter blocks, command timeouts, unknown tools.
- **Concurrency tests**: concurrent `ConversationMemory.add`, concurrent
  `LongTermMemory.save` to same key.
- **Test count**: 109 passing, 2 skipped (platform-specific symlink tests).
- **Command**: `pytest -v`

## 12. Observability

### Structured logging
`StructuredLogger` emits JSON-lines to stderr and optionally to a file.
Each event includes:
- `timestamp` (ISO 8601, UTC)
- `level` (INFO, WARN, ERROR)
- `event` (semantic event name)
- `correlation_id` (UUID per task)
- Domain-specific fields (tool name, error details, metrics)

Key events:
- `task_started` — task received, correlation ID assigned
- `tool_call` — tool name and correlation ID logged
- `tool_permission_denied` — blocked tool logged with error
- `tool_error` — tool execution failure logged
- `api_retry` — retryable API error logged with attempt count
- `api_non_retryable_error` — non-retryable error logged
- `safety_filter_blocked` — empty candidates detected
- `task_completed` — final metrics and text length

### Metrics
`Metrics` tracks in-process counters:
- `tokens_used` — (placeholder for future token counting)
- `latency_ms` — total task latency
- `tool_calls` — number of tool invocations
- `errors` — tool errors and permission denials

### What operators can observe
- Whether a task is running (via log stream).
- Which tools the agent is calling.
- Retry behavior and backoff delays.
- Safety filter blocks.
- Error rates and tool call counts.

## 13. Deployment

### Local development
```bash
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your Gemini API key
python -m gemini_agent_toolkit --directory . --task "Summarize README.md"
```

### Interactive mode
```bash
python main.py
```

### CI
GitHub Actions runs on every push and PR:
1. `ruff check` — linting
2. `mypy --follow-imports=skip` — type checking
3. `pytest -v` — full test suite

## 14. Known limitations

1. **Single-agent only**: The `Agent` class is not thread-safe. It is
   designed for sequential task execution in a single process.
2. **File-based storage**: `LongTermMemory` uses JSON files, not a
   database. Not suitable for high-throughput or multi-process
   scenarios.
3. **No multi-tenancy**: The agent runs with a single API key and has no
   concept of users or tenants.
4. **`google.generativeai` deprecation**: The `google-generativeai`
   package is deprecated by Google in favor of `google.genai`. Migration
   is future work.
5. **Symlink tests skipped on Windows**: Tests that create symlinks
   require administrator privileges on Windows and are skipped.
6. **`echo` unavailable with `shell=False`**: Shell built-ins like
   `echo` cannot be executed because the command runner uses
   `shell=False` for security. Use OS executables instead.

## 15. Future work

1. Migrate from `google.generativeai` to `google.genai` (new SDK).
2. Add token counting to `Metrics` (parse `usage_metadata` from
   Gemini responses).
3. Add a `--verbose` / `--quiet` flag to control log level.
4. Add integration tests that verify end-to-end behavior with a mock
   Gemini server (e.g., using `respx` or `responses`).
5. Add an optional `--allowed-tools` CLI argument to restrict tool
   access from the command line.
