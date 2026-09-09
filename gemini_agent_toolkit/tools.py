import os
import shlex
import subprocess
from pathlib import Path

from gemini_agent_toolkit.config import settings


def get_base_dir() -> str:
    """Return the real working directory, resolved against symlinks.

    Computed at call-time rather than import-time so that ``os.chdir``
    (e.g. in ``__main__.py``) does not leave a stale base directory.
    """
    return os.path.realpath(os.getcwd())


# Commands that are too dangerous to ever execute, even inside the sandbox.
BLOCKED_COMMANDS = frozenset({
    "rm", "rmdir", "del", "format", "fdisk", "mkfs",
    "chmod", "chown", "icacls",
    "curl", "wget", "nc", "ncat", "netcat",
    "python", "python3", "pip", "pip3",
    "git", "svn", "hg",
    "sudo", "su", "runas",
    "shutdown", "reboot", "halt",
    "taskkill", "kill", "pkill",
})


# -------- SECURITY HELPERS -------- #

def is_safe_path(path: str, base_dir: str | None = None) -> bool:
    """Ensure *path* resolves inside *base_dir* (default: current working dir).

    Uses ``os.path.realpath`` to resolve symlinks and ``..`` components,
    then ``os.path.commonpath`` to avoid the prefix-attack that ``str.startswith``
    is vulnerable to (e.g. ``/safe/dir_evil`` vs ``/safe/dir``).
    """
    base = os.path.realpath(base_dir) if base_dir else get_base_dir()
    try:
        real = os.path.realpath(os.path.join(base, path))
    except (ValueError, OSError):
        return False
    try:
        return os.path.commonpath([base, real]) == base
    except ValueError:
        return False


def sanitize_path(path: str, base_dir: str | None = None) -> str:
    """Resolve *path* and verify it stays inside the working directory.

    Raises ``ValueError`` if the path is absolute, contains traversal
    (``..``), or escapes the sandbox after symlink resolution.
    """
    if not path or not isinstance(path, str):
        raise ValueError("Path must be a non-empty string.")

    if os.path.isabs(path):
        raise ValueError("Absolute paths are not allowed.")

    if ".." in path.split(os.path.sep) or ".." in path.split("/"):
        raise ValueError("Path traversal ('..') is not allowed.")

    base = os.path.realpath(base_dir) if base_dir else get_base_dir()

    try:
        real = os.path.realpath(os.path.join(base, path))
    except (ValueError, OSError) as exc:
        raise ValueError(f"Unable to resolve path: {exc}") from exc

    try:
        if os.path.commonpath([base, real]) != base:
            raise ValueError("Blocked: Path escapes the allowed directory.")
    except ValueError:
        raise ValueError("Blocked: Path escapes the allowed directory.")

    return real


# -------- FILE FUNCTIONS -------- #

def read_file(file_path: str) -> str:
    """Reads the content of a file inside the allowed directory."""
    try:
        safe_path = sanitize_path(file_path)
        with open(safe_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(file_path: str, content: str) -> str:
    """Writes content to a file inside the allowed directory."""
    try:
        safe_path = sanitize_path(file_path)
        Path(os.path.dirname(safe_path)).mkdir(parents=True, exist_ok=True)
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File '{file_path}' written successfully."
    except Exception as e:
        return f"Error writing file: {e}"


# -------- COMMAND EXECUTION -------- #

def execute_command(command: str) -> str:
    """Executes a tokenized shell command safely.

    * ``shell=False`` — no shell interpretation, no glob expansion, no
      metacharacter injection.
    * Commands are checked against ``BLOCKED_COMMANDS``.
    * The working directory is the real (symlink-resolved) cwd.
    """
    try:
        parts = shlex.split(command)
        if not parts:
            return "Error: Empty command."

        # Reject any argument that attempts path traversal.
        for p in parts:
            if ".." in p.split(os.path.sep) or ".." in p.split("/"):
                return f"Command blocked: '{p}' contains path traversal."

        # Block dangerous commands.
        if parts[0] in BLOCKED_COMMANDS:
            return f"Command blocked: '{parts[0]}' is not allowed for security."

        # Reject shell metacharacters in any argument.
        dangerous_chars = set(";|&`$(){}<>\\\n\r")
        for p in parts:
            if any(c in p for c in dangerous_chars):
                return f"Command blocked: '{p}' contains shell metacharacters."

        base = get_base_dir()
        result = subprocess.run(
            parts,
            shell=False,
            cwd=base,
            capture_output=True,
            text=True,
            timeout=settings.command_timeout_seconds,
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if stdout and stderr:
            return f"Stdout:\n{stdout}\n\nStderr:\n{stderr}"
        if stdout:
            return f"Stdout:\n{stdout}"
        if stderr:
            return f"Stderr:\n{stderr}"
        return "Command executed successfully with no output."

    except subprocess.TimeoutExpired:
        return (
            f"Error: Command timed out after {settings.command_timeout_seconds} seconds."
        )
    except FileNotFoundError:
        return f"Error: Command not found: {parts[0]}"
    except Exception as e:
        return f"Error executing command: {e}"
