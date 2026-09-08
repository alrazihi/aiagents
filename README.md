# Gemini Agent Toolkit

A lightweight Python toolkit for building agentic workflows with Google Gemini. It provides composable building blocks for tool use, memory, guardrails, observability, and evaluation.

## Features

- **Tool calling**: read/write files and execute shell commands in a restricted working directory
- **Memory**: conversation history with configurable turn limits
- **Guardrails**: input length checks, blocked-pattern redaction, and tool allowlists
- **Observability**: structured JSON logging with optional file output
- **Evaluation**: task-case harness for measuring tool accuracy and latency

## Quick Start

```bash
pip install -r dev-requirements.txt
cp .env.example .env
python main.py --directory . --task "Summarize README.md"
```

## Example

```python
from gemini_agent_toolkit.agent import Agent
from gemini_agent_toolkit.memory import ConversationMemory
from gemini_agent_toolkit.guardrails import Guardrails

memory = ConversationMemory(max_turns=10)
guardrails = Guardrails(allowed_tools=["read_file", "write_file"])

agent = Agent(api_key="...")
agent.run_task("Read README.md and summarize it")
```

## Testing

```bash
pytest
```

## Architecture

```
gemini_agent_toolkit/
  agent.py           # Gemini client with retry/backoff
  tools.py           # File and command tools with path sandboxing
  memory.py          # Conversation memory
  guardrails.py      # Input/output validation
  observability.py   # Structured logging
  evaluator.py       # Task-case evaluation harness
  config.py          # Settings via pydantic-settings
```

## License

MIT
