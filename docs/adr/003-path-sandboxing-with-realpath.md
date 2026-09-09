# ADR-003: Path sandboxing with realpath and commonpath

## Status

Implemented

## Context

The original path-sandboxing used two vulnerable mechanisms:

1. `BASE_DIR = os.path.abspath(os.getcwd())` — captured at import time.
   If `__main__.py` calls `os.chdir(args.directory)`, the base directory
   becomes stale and the new working directory is unreachable.

2. `real.startswith(BASE_DIR)` — vulnerable to a **prefix attack**:
   if `BASE_DIR = /home/user/project`, a path resolving to
   `/home/user/project_evil/secret` also `startswith` `/home/user/project`,
   bypassing the sandbox.

3. `os.path.abspath` does not resolve symlinks. A symlink inside the
   sandbox pointing to `/etc/passwd` would bypass the check.

## Decision

1. Replace `BASE_DIR` constant with `get_base_dir()` that calls
   `os.path.realpath(os.getcwd())` at call-time, resolving symlinks and
   reflecting `os.chdir` calls.

2. In `sanitize_path` and `is_safe_path`, use `os.path.realpath` to
   resolve the full path (including symlinks and `..` components), then
   use `os.path.commonpath([base, real]) == base` instead of
   `str.startswith` to prevent prefix attacks.

3. Reject absolute paths and `..` components as a first line of
   defense before the `realpath` check.

## Consequences

- The sandbox is now correct under `os.chdir`, symlinks, and prefix
  attacks.
- A path like `subdir/../../etc/passwd` is rejected both by the `..`
  check and by `commonpath`.
- Symlink-based escapes are caught because `realpath` resolves the
  final target, not the link itself.

## Evidence

- `gemini_agent_toolkit/tools.py:7-76` — `get_base_dir`, `is_safe_path`,
  `sanitize_path`
- `tests/test_tools.py::test_is_safe_path_prefix_attack`
- `tests/test_tools.py::test_sanitize_path_blocks_windows_drive_path`
- `tests/test_tools.py::test_sanitize_path_blocks_backslash_traversal`
- `tests/test_tools.py::test_get_base_dir_resolves_symlinks`
