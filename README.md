# Gemini Agent Toolkit

A lightweight Python toolkit for building agentic workflows with Google Gemini.
It provides composable building blocks for tool use, memory, guardrails,
observability, and evaluation.

## Features

- **Tool calling**: read/write files and execute shell commands in a restricted
  working directory (path sandboxing with `realpath`+`commonpath`,
  `shell=False` execution, command block-list, 30s timeout)
- **Memory**: conversation history with configurable turn limits,
  thread-safe; plus file-backed long-term storage with atomic writes
- **Guardrails**: case-insensitive input length checks, blocked-pattern
  output redaction, and tool allow-list enforcement — all integrated into
  the agent execution loop
- **Observability**: structured JSON logging with optional file output,
  correlation IDs, and in-process metrics (tool calls, errors, latency)
- **Evaluation**: task-case harness for measuring tool accuracy and latency
  with percentile reporting

## Quick Start

```bash
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your Gemini API key
python main.py --directory . --task "Summarize README.md"
```

## CLI Usage

```bash
# Run a single task:
python main.py --directory . --task "Summarize README.md"

# Run a health check:
python main.py --health-check

# Interactive REPL:
python main.py --directory .
```

## Library Usage

```python
from gemini_agent_toolkit.agent import Agent

agent = Agent(
    api_key="...",
    allowed_tools=["read_file", "write_file"],  # tool allow-list
)
result = agent.run_task("Read README.md and summarize it")
print(result)
```

## Testing

```bash
pytest -v
```

132 tests pass (2 skipped on Windows for symlink permissions). Tests cover
unit behavior, security boundaries, failure paths, concurrency, and
end-to-end agent flows with mocked Gemini responses.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full architecture document.
Architecture Decision Records are in [docs/adr/](docs/adr/).

```
gemini_agent_toolkit/
  agent.py           # LLM orchestration: retry/backoff, tool loop, guardrails
  tools.py           # File and command tools with path sandboxing
  memory.py          # Conversation memory (thread-safe) + long-term storage
  guardrails.py      # Input/output validation + tool allow-listing
  observability.py   # Structured JSON logging + metrics
  evaluator.py       # Task-case evaluation harness
  config.py          # Settings via pydantic-settings
```

## License

MIT
