# ADR-006: Case-insensitive guardrail patterns

## Status

Implemented

## Context

The original blocked-pattern regexes were case-sensitive:

```python
re.compile(r"rm -rf /")
re.compile(r"DROP TABLE")
```

A model could trivially bypass these by using different casing:
`rm -RF /`, `drop table users`, `Delete From users Where 1=1`.

## Decision

1. Add `re.IGNORECASE` to all pattern compilations.
2. Expand the pattern set to cover more dangerous commands:
   - `rm -rf .*` (wider match for recursive delete)
   - `DROP COLUMN`, `TRUNCATE TABLE` (additional SQL)
   - `> /dev/sd` (disk overwrite)
   - `format [drive]:` (disk format)
   - `chmod -r` (permission escalation)

## Consequences

- `DROP TABLE`, `drop table`, `Drop Table` are all caught.
- `rm -rf /`, `RM -RF /`, `Rm -Rf /` are all caught.
- Output redaction operates on the LLM-generated text, providing
  defense-in-depth alongside input validation and command blocking.

## Evidence

- `gemini_agent_toolkit/guardrails.py:14-25` — case-insensitive patterns
- `tests/test_guardrails.py::test_validate_output_case_insensitive`
