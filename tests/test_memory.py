import threading

from gemini_agent_toolkit.memory import ConversationMemory, LongTermMemory, Message

# -------- ConversationMemory (existing + expanded) -------- #

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


def test_message_stores_timestamp_and_metadata():
    from datetime import datetime
    msg = Message(role="user", content="test", metadata={"key": "value"})
    d = msg.to_dict()
    assert d["role"] == "user"
    assert d["content"] == "test"
    assert d["metadata"] == {"key": "value"}
    assert "timestamp" in d
    ts = datetime.fromisoformat(d["timestamp"])
    assert ts.tzinfo is not None


def test_conversation_memory_clear():
    memory = ConversationMemory(max_turns=5)
    memory.add("user", "Hello")
    assert len(memory.get_history()) == 1
    memory.clear()
    assert len(memory.get_history()) == 0


def test_conversation_memory_metadata_added():
    memory = ConversationMemory(max_turns=5)
    memory.add("user", "Hello", metadata={"source": "test"})
    history = memory.get_history()
    assert history[0]["metadata"] == {"source": "test"}


# -------- LongTermMemory -------- #

def test_longterm_memory_save_and_load(tmp_path):
    mem = LongTermMemory(storage_path=str(tmp_path / ".memory"))
    mem.save("key1", {"value": "hello"})
    loaded = mem.load("key1")
    assert loaded == {"value": "hello"}


def test_longterm_memory_load_nonexistent(tmp_path):
    mem = LongTermMemory(storage_path=str(tmp_path / ".memory"))
    assert mem.load("nonexistent") is None


def test_longterm_memory_overwrite(tmp_path):
    mem = LongTermMemory(storage_path=str(tmp_path / ".memory"))
    mem.save("key1", {"value": "old"})
    mem.save("key1", {"value": "new"})
    loaded = mem.load("key1")
    assert loaded == {"value": "new"}


def test_longterm_memory_creates_storage_dir(tmp_path):
    storage = tmp_path / ".memory"
    assert not storage.exists()
    mem = LongTermMemory(storage_path=str(storage))
    mem.save("key1", {"value": "hello"})
    assert storage.exists()


# -------- Concurrency tests -------- #

def test_conversation_memory_thread_safe_concurrent_adds():
    """Concurrent add() calls must not lose messages or crash."""
    memory = ConversationMemory(max_turns=1000)
    num_threads = 10
    messages_per_thread = 50

    def worker():
        for i in range(messages_per_thread):
            memory.add("user", f"msg-{i}")

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    history = memory.get_history()
    expected = num_threads * messages_per_thread
    assert len(history) == expected, f"Expected {expected} messages, got {len(history)}"


def test_conversation_memory_thread_safe_concurrent_add_and_get():
    """Concurrent get_history() during add() must not crash or return partial data."""
    memory = ConversationMemory(max_turns=10000)
    errors = []

    def writer():
        try:
            for i in range(200):
                memory.add("user", f"write-{i}")
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for _ in range(200):
                memory.get_history()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"Concurrent access errors: {errors}"


def test_longterm_memory_atomic_write_no_corruption(tmp_path):
    """Concurrent saves to the same key must not produce corrupted JSON."""
    mem = LongTermMemory(storage_path=str(tmp_path / ".memory"))
    num_threads = 10

    def worker(tid):
        for i in range(10):
            mem.save("shared_key", {"thread": tid, "seq": i})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    loaded = mem.load("shared_key")
    assert loaded is not None
    assert isinstance(loaded, dict)
    assert "thread" in loaded
    assert "seq" in loaded
