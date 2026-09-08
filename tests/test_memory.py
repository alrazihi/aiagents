import pytest
from gemini_agent_toolkit.memory import ConversationMemory


def test_add_and_get_history():
    memory = ConversationMemory(max_turns=2)
    memory.add("user", "Hello")
    memory.add("assistant", "Hi there")
    history = memory.get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_max_turns_trims_history():
    memory = ConversationMemory(max_turns=1)
    memory.add("user", "First")
    memory.add("assistant", "Second")
    memory.add("user", "Third")
    history = memory.get_history()
    assert len(history) == 2
    assert history[0]["content"] == "Second"
    assert history[1]["content"] == "Third"


def test_empty_history():
    memory = ConversationMemory()
    assert memory.get_history() == []
