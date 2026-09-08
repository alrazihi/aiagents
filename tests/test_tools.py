import os
import pytest
from gemini_agent_toolkit.tools import is_safe_path, sanitize_path, read_file, write_file


def test_is_safe_path_blocks_traversal():
    assert not is_safe_path("..\\secret.txt")


def test_sanitize_path_blocks_absolute_paths():
    with pytest.raises(ValueError):
        sanitize_path("/etc/passwd")


def test_sanitize_path_blocks_parent_traversal():
    with pytest.raises(ValueError):
        sanitize_path("..\\..\\secret.txt")


def test_read_and_write_file(tmp_path, monkeypatch):
    test_file = tmp_path / "test.txt"
    monkeypatch.chdir(tmp_path)
    msg = write_file("test.txt", "hello")
    assert "successfully" in msg
    content = read_file("test.txt")
    assert content == "hello"
