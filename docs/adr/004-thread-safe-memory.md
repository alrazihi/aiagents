# ADR-004: Thread-safe memory with atomic file writes

## Status

Implemented

## Context

The original `ConversationMemory` used a plain `list` with no
synchronization. The `add` method performed a non-atomic read-trim-write
sequence:

```python
self.messages.append(msg)
if len(self.messages) > self.max_turns * 2:
    self.messages = self.messages[-self.max_turns * 2 :]
```

Under concurrent access, two threads could both trim the list based on a
stale length, losing messages. `LongTermMemory.save` wrote directly to
the target file with `Path.write_text`, producing partial/corrupted JSON
if the process was killed mid-write or if two threads wrote the same key
simultaneously (especially on Windows where `os.replace` is not
concurrent-safe).

## Decision

1. Add a `threading.Lock` to `ConversationMemory` and guard all
   `add`, `get_history`, and `clear` operations with it.

2. In `LongTermMemory.save`, write to a temporary file via
   `tempfile.mkstemp`, then use `os.replace` (atomic on both POSIX and
   Windows) to swap it into place. Add a `threading.Lock` to serialise
   writes to the same key, preventing `PermissionError` on Windows when
   two threads race on `os.replace`.

3. Clean up the temp file in a `finally`-style `except BaseException`
   block to prevent temp file accumulation.

## Consequences

- Concurrent `add` calls cannot lose messages or observe half-trimmed
  lists.
- `save` is safe against partial writes and process kills — either the
  old file or the new file is fully visible.
- A single `LongTermMemory` instance serialises all writes, which is
  acceptable for a toolkit library. High-throughput multi-process
  scenarios would require a database.

## Evidence

- `gemini_agent_toolkit/memory.py:42-64` — locked `ConversationMemory`
- `gemini_agent_toolkit/memory.py:76-97` — atomic `LongTermMemory.save`
- `tests/test_memory.py::test_conversation_memory_thread_safe_concurrent_adds`
- `tests/test_memory.py::test_conversation_memory_thread_safe_concurrent_add_and_get`
- `tests/test_memory.py::test_longterm_memory_atomic_write_no_corruption`
