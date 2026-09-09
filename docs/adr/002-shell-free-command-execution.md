# ADR-002: Shell-free command execution

## Status

Implemented

## Context

The original `execute_command` used `subprocess.run(command, shell=True)`,
which passes the command through the system shell. This allows an attacker
(or a compromised LLM) to inject shell metacharacters:

```
ls; rm -rf /
echo $(cat /etc/passwd)
echo `whoami`
```

Even within a path sandbox, shell injection enables arbitrary command
execution, environment variable expansion, and process substitution.

## Decision

1. Replace `shell=True` with `shell=False` and `shlex.split(command)`.
   This tokenizes the command without shell interpretation.
2. Block dangerous commands (`rm`, `curl`, `wget`, `format`, `chmod`,
   `sudo`, etc.) via a `BLOCKED_COMMANDS` frozenset.
3. Reject any argument containing shell metacharacters
   (`;`, `|`, `&`, `` ` ``, `$`, `()`, `{}``, `<>`, `\`, newlines).
4. Add a 30-second timeout (`subprocess.TimeoutExpired` is caught and
   reported).

## Consequences

- Shell built-ins (`echo`, `cd`, `dir`) are no longer available because
  they require `shell=True`. Only OS executables can be run.
- Metacharacter injection is eliminated at the tokenizer level.
- The command allow/deny list is explicit and auditable.
- On Windows, `shell=False` is stricter; users must call executables
  directly (e.g. `where` instead of `echo`).

## Evidence

- `gemini_agent_toolkit/tools.py:55-100` — `execute_command` with `shell=False`
- `tests/test_tools.py::test_execute_command_blocks_shell_metacharacters`
- `tests/test_tools.py::test_execute_command_blocks_dangerous_commands`
- `tests/test_tools.py::test_execute_command_timeout`
