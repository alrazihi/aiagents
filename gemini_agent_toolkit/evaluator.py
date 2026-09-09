"""Evaluation harness for agent task completion and tool accuracy."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskCase:
    name: str
    prompt: str
    expected_tools: list[str] = field(default_factory=list)
    validator: Callable[[Any], bool] | None = None


@dataclass
class TaskResult:
    task: TaskCase
    success: bool
    tool_accuracy: float
    latency_ms: float
    tokens_used: int | None = None
    error: str | None = None


class EvaluationHarness:
    def __init__(self):
        self.results: list[TaskResult] = []

    def run(self, agent, cases: list[TaskCase]) -> dict[str, float]:
        self.results = []
        for case in cases:
            start = time.perf_counter()
            try:
                response = agent.run_task(case.prompt)
                latency = (time.perf_counter() - start) * 1000
                if case.validator is not None:
                    success = bool(case.validator(response))
                else:
                    success = True
                tool_accuracy = self._compute_tool_accuracy(case, response)
                self.results.append(TaskResult(
                    task=case, success=success, tool_accuracy=tool_accuracy,
                    latency_ms=latency,
                ))
            except Exception as exc:
                latency = (time.perf_counter() - start) * 1000
                self.results.append(TaskResult(
                    task=case, success=False, tool_accuracy=0.0,
                    latency_ms=latency, error=str(exc),
                ))

        return self._summary()

    def _compute_tool_accuracy(self, case: TaskCase, response: str) -> float:
        if not case.expected_tools:
            return 1.0
        hits = sum(1 for tool in case.expected_tools if tool in (response or ""))
        return hits / len(case.expected_tools)

    def _summary(self) -> dict[str, float]:
        if not self.results:
            return {}
        success_rate = sum(1 for r in self.results if r.success) / len(self.results)
        tool_accuracy = statistics.mean(r.tool_accuracy for r in self.results)
        latency_p95 = self._percentile([r.latency_ms for r in self.results], 95)
        return {
            "success_rate": round(success_rate, 4),
            "tool_accuracy": round(tool_accuracy, 4),
            "latency_p95_ms": round(latency_p95, 2),
        }

    @staticmethod
    def _percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        k = (len(values) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(values):
            return values[f]
        return values[f] + (k - f) * (values[c] - values[f])
