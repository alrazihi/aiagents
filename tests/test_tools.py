import os
import sys
from unittest.mock import patch

import pytest

from gemini_agent_toolkit.tools import (
    _allowed_commands,
    execute_command,
    get_base_dir,
    is_safe_path,
    read_file,
    sanitize_path,
    write_file,
)

# -------- Path safety tests (existing) -------- #

def test_is_safe_path_blocks_traversal():
    assert not is_safe_path("..\\secret.txt")


def test_sanitize_path_blocks_absolute_paths():
    with pytest.raises(ValueError):
        sanitize_path("/etc/passwd")


def test_sanitize_path_blocks_parent_traversal():
    with pytest.raises(ValueError):
        sanitize_path("..\\..\\secret.txt")


def test_read_and_write_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    msg = write_file("test.txt", "hello")
    assert "successfully" in msg
    content = read_file("test.txt")
    assert content == "hello"


# -------- Path safety: symlink resolution -------- #

@pytest.mark.skipif(sys.platform == "win32",
                    reason="Symlink creation may require admin on Windows")
def test_is_safe_path_resolves_symlinks(tmp_path, monkeypatch):
    """A symlink inside the sandbox pointing outside must be blocked."""
    outside = tmp_path / ".." / "outside_secret.txt"
    outside.write_text("secret")
    link = tmp_path / "link_to_outside"
    link.symlink_to(outside.resolve())
    monkeypatch.chdir(tmp_path)
    assert not is_safe_path("link_to_outside")


@pytest.mark.skipif(sys.platform == "win32",
                    reason="Symlink creation may require admin on Windows")
def test_sanitize_path_blocks_symlink_escape(tmp_path, monkeypatch):
    """sanitize_path must reject a symlink that resolves outside base_dir."""
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret")
    link = tmp_path / "escape_link"
    link.symlink_to(outside.resolve())
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        sanitize_path("escape_link")


def test_sanitize_path_blocks_empty_string():
    """Empty string is not a valid path."""
    with pytest.raises(ValueError):
        sanitize_path("")


def test_sanitize_path_blocks_none():
    """None is not a valid path."""
    with pytest.raises((ValueError, TypeError)):
        sanitize_path(None)


def test_is_safe_path_prefix_attack(tmp_path, monkeypatch):
    """str.startswith prefix attack: /base_dir_evil must not match /base_dir."""
    monkeypatch.chdir(tmp_path)
    sibling = tmp_path.parent / (tmp_path.name + "_evil")
    sibling.mkdir(exist_ok=True)
    evil_path = os.path.join(str(sibling), "secret.txt")
    assert not is_safe_path(evil_path)


# -------- Path safety: Windows paths -------- #

def test_sanitize_path_blocks_windows_drive_path():
    with pytest.raises(ValueError):
        sanitize_path("C:\\Windows\\system32")


def test_sanitize_path_blocks_backslash_traversal():
    with pytest.raises(ValueError):
        sanitize_path("..\\..\\..\\etc\\passwd")


# -------- File read/write edge cases -------- #

def test_read_file_nonexistent_returns_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = read_file("nonexistent.txt")
    assert "Error" in result


def test_write_file_creates_parent_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    msg = write_file("subdir/nested/file.txt", "content")
    assert "successfully" in msg
    assert (tmp_path / "subdir" / "nested" / "file.txt").read_text() == "content"


def test_read_file_non_utf8_returns_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    binary_path = tmp_path / "data.bin"
    binary_path.write_bytes(b"\x80\x81\x82\xff")
    result = read_file("data.bin")
    assert "Error" in result


# -------- Command execution security tests -------- #

def test_execute_command_blocks_dangerous_commands(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for cmd in ["rm -rf /", "format c:", "curl http://evil.com", "wget http://evil.com"]:
        result = execute_command(cmd)
        assert (
            "blocked" in result.lower()
            or "not allowed" in result.lower()
        ), f"Command not blocked: {cmd}"


def test_execute_command_blocks_shell_metacharacters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for cmd in ["echo hello; rm -rf /", "echo hello | cat", "echo $(whoami)", "echo `whoami`"]:
        result = execute_command(cmd)
        assert "blocked" in result.lower(), f"Metacharacter not blocked: {cmd}"


def test_execute_command_empty_command():
    result = execute_command("")
    assert "Error" in result or "Empty" in result


def test_execute_command_not_in_allowlist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = execute_command("nonexistent_command_xyz123")
    assert "not in the allowed list" in result.lower() or "blocked" in result.lower()


def test_execute_command_allows_safe_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    if sys.platform == "win32":
        result = execute_command("where where")
    else:
        result = execute_command("echo hello_world")
    assert "hello_world" in result or "where" in result.lower()


def test_execute_command_timeout(tmp_path, monkeypatch):
    """Commands that timeout should return a timeout error, not hang forever."""
    import subprocess
    monkeypatch.chdir(tmp_path)
    with patch("gemini_agent_toolkit.tools.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep", timeout=5)
        result = execute_command("sleep 10")
    assert "timeout" in result.lower() or "timed out" in result.lower()


def test_allowed_commands_list_is_frozen():
    assert isinstance(_allowed_commands(), frozenset)
    assert "ls" in _allowed_commands()


def test_execute_command_blocks_disallowed_command():
    result = execute_command("python --version")
    assert "blocked" in result.lower()


def test_get_base_dir_resolves_symlinks(tmp_path, monkeypatch):
    """get_base_dir must return the realpath, not the logical path."""
    monkeypatch.chdir(tmp_path)
    base = get_base_dir()
    assert base == os.path.realpath(str(tmp_path))


def test_sanitize_path_returns_realpath(tmp_path, monkeypatch):
    """sanitize_path must return a realpath that has no symlinks."""
    monkeypatch.chdir(tmp_path)
    result = sanitize_path("test.txt")
    assert result == os.path.realpath(os.path.join(str(tmp_path), "test.txt"))
