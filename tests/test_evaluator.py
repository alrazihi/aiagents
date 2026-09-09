import pytest

from gemini_agent_toolkit.evaluator import EvaluationHarness, TaskCase, TaskResult

# -------- TaskCase defaults -------- #

def test_task_case_defaults():
    tc = TaskCase(name="test", prompt="hello")
    assert tc.name == "test"
    assert tc.prompt == "hello"
    assert tc.expected_tools == []
    assert tc.validator is None


def test_task_case_with_expected_tools():
    tc = TaskCase(name="test", prompt="hello", expected_tools=["read_file", "write_file"])
    assert tc.expected_tools == ["read_file", "write_file"]


# -------- TaskResult defaults -------- #

def test_task_result_defaults():
    tc = TaskCase(name="test", prompt="hello")
    tr = TaskResult(task=tc, success=True, tool_accuracy=1.0, latency_ms=100.0)
    assert tr.tokens_used is None
    assert tr.error is None


# -------- Tool accuracy calculation -------- #

def test_compute_tool_accuracy_no_expected_tools():
    harness = EvaluationHarness()
    tc = TaskCase(name="test", prompt="hello")
    accuracy = harness._compute_tool_accuracy(tc, "some response text")
    assert accuracy == 1.0


def test_compute_tool_accuracy_all_hits():
    harness = EvaluationHarness()
    tc = TaskCase(name="test", prompt="hello", expected_tools=["read_file", "write_file"])
    accuracy = harness._compute_tool_accuracy(tc, "I used read_file and write_file")
    assert accuracy == 1.0


def test_compute_tool_accuracy_partial_hits():
    harness = EvaluationHarness()
    tc = TaskCase(
        name="test",
        prompt="hello",
        expected_tools=["read_file", "write_file", "execute_command"],
    )
    accuracy = harness._compute_tool_accuracy(tc, "I used read_file")
    assert accuracy == pytest.approx(1 / 3, abs=0.01)


def test_compute_tool_accuracy_none_hits():
    harness = EvaluationHarness()
    tc = TaskCase(name="test", prompt="hello", expected_tools=["read_file", "write_file"])
    accuracy = harness._compute_tool_accuracy(tc, "I did nothing")
    assert accuracy == 0.0


def test_compute_tool_accuracy_none_response():
    harness = EvaluationHarness()
    tc = TaskCase(name="test", prompt="hello", expected_tools=["read_file"])
    accuracy = harness._compute_tool_accuracy(tc, None)
    assert accuracy == 0.0


# -------- Percentile calculation -------- #

def test_percentile_single_value():
    assert EvaluationHarness._percentile([100.0], 95) == 100.0


def test_percentile_empty():
    assert EvaluationHarness._percentile([], 95) == 0.0


def test_percentile_p95_known_values():
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    result = EvaluationHarness._percentile(values, 95)
    assert result == pytest.approx(95.5, abs=0.01)


def test_percentile_minimum():
    values = [10.0, 20.0, 30.0]
    assert EvaluationHarness._percentile(values, 0) == 10.0


def test_percentile_maximum():
    values = [10.0, 20.0, 30.0]
    assert EvaluationHarness._percentile(values, 100) == 30.0


# -------- Summary calculation -------- #

def test_summary_empty_results():
    harness = EvaluationHarness()
    assert harness._summary() == {}


def test_summary_success_rate():
    harness = EvaluationHarness()
    tc = TaskCase(name="test", prompt="hello")
    harness.results = [
        TaskResult(task=tc, success=True, tool_accuracy=1.0, latency_ms=100.0),
        TaskResult(task=tc, success=True, tool_accuracy=0.5, latency_ms=200.0),
        TaskResult(task=tc, success=False, tool_accuracy=0.0, latency_ms=150.0),
        TaskResult(task=tc, success=True, tool_accuracy=0.5, latency_ms=120.0),
    ]
    summary = harness._summary()
    assert summary["success_rate"] == 0.75
    assert summary["tool_accuracy"] == pytest.approx(0.5, abs=0.01)


# -------- Run harness -------- #

class FakeAgent:
    """Minimal agent stub that returns a fixed response."""

    def __init__(self, response="task completed successfully"):
        self._response = response
        self.call_count = 0

    def run_task(self, prompt: str) -> str:
        self.call_count += 1
        if prompt == "fail":
            raise RuntimeError("agent failed")
        return self._response


def test_run_with_validator_pass():
    harness = EvaluationHarness()
    tc = TaskCase(
        name="test",
        prompt="complete the task",
        validator=lambda response: "success" in response,
    )
    summary = harness.run(FakeAgent("task completed successfully"), [tc])
    assert summary["success_rate"] == 1.0
    assert harness.results[0].success is True


def test_run_with_validator_fail():
    harness = EvaluationHarness()
    tc = TaskCase(
        name="test",
        prompt="complete the task",
        validator=lambda response: "expected_substring" in response,
    )
    summary = harness.run(FakeAgent("nothing useful here"), [tc])
    assert summary["success_rate"] == 0.0
    assert harness.results[0].success is False


def test_run_no_validator():
    harness = EvaluationHarness()
    tc = TaskCase(name="test", prompt="complete the task")
    summary = harness.run(FakeAgent("done"), [tc])
    assert summary["success_rate"] == 1.0


def test_run_handles_exception():
    harness = EvaluationHarness()
    tc = TaskCase(name="test", prompt="fail")
    summary = harness.run(FakeAgent("won't be called"), [tc])
    assert summary["success_rate"] == 0.0
    assert harness.results[0].success is False
    assert harness.results[0].error is not None


def test_run_records_latency():
    harness = EvaluationHarness()
    tc = TaskCase(name="test", prompt="ok")
    harness.run(FakeAgent("done"), [tc])
    assert harness.results[0].latency_ms >= 0.0


def test_run_multiple_cases():
    harness = EvaluationHarness()
    cases = [
        TaskCase(name="t1", prompt="ok", validator=lambda r: "yes" in r),
        TaskCase(name="t2", prompt="ok", validator=lambda r: "yes" in r),
    ]
    summary = harness.run(FakeAgent("yes it is"), cases)
    assert summary["success_rate"] == 1.0
    assert len(harness.results) == 2
