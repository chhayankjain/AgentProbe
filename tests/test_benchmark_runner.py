"""Tests for BenchmarkRunner — multi-run experiment orchestration."""

from __future__ import annotations

import pytest

from agentprobe.benchmark.runner import BenchmarkConfig, BenchmarkRunner, RunResult
from agentprobe.injector.tool_failure import ToolFailureConfig, ToolFailureType


# ---------------------------------------------------------------------------
# Mock agents
# ---------------------------------------------------------------------------


class AlwaysSuccessAgent:
    framework = "mock"

    def invoke(self, input: dict) -> dict:
        return {"output": "mock output", "latency_ms": 10.0}


class AlwaysFailAgent:
    framework = "mock"

    def invoke(self, input: dict) -> dict:
        raise RuntimeError("Mock agent failure")


class FlipFlopAgent:
    """Alternates success / failure each call."""
    framework = "mock"

    def __init__(self):
        self._count = 0

    def invoke(self, input: dict) -> dict:
        self._count += 1
        if self._count % 2 == 0:
            raise RuntimeError("Even call failure")
        return {"output": "ok", "latency_ms": 5.0}


class CustomOutputAgent:
    framework = "mock"

    def __init__(self, output: str = "custom output"):
        self._output = output

    def invoke(self, input: dict) -> dict:
        return {"output": self._output, "latency_ms": 25.0}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBenchmarkRunner:
    def _config(self, n_runs: int = 5, recovery: int = 0, **kwargs) -> BenchmarkConfig:
        return BenchmarkConfig(
            framework="mock",
            task="tool_use",
            n_runs=n_runs,
            recovery_attempts=recovery,
            **kwargs,
        )

    def test_all_success_all_runs_succeed(self):
        runner = BenchmarkRunner(AlwaysSuccessAgent(), self._config(n_runs=5))
        results = runner.run()
        assert len(results) == 5
        assert all(r.success for r in results)

    def test_all_failure_all_runs_fail(self):
        runner = BenchmarkRunner(AlwaysFailAgent(), self._config(n_runs=5))
        results = runner.run()
        assert len(results) == 5
        assert all(not r.success for r in results)

    def test_flip_flop_exactly_half_fail(self):
        runner = BenchmarkRunner(FlipFlopAgent(), self._config(n_runs=10))
        results = runner.run()
        successes = sum(1 for r in results if r.success)
        failures = sum(1 for r in results if not r.success)
        assert successes == 5
        assert failures == 5

    def test_correct_run_count(self):
        for n in [1, 10, 25]:
            runner = BenchmarkRunner(AlwaysSuccessAgent(), self._config(n_runs=n))
            results = runner.run()
            assert len(results) == n

    def test_run_result_framework_and_task(self):
        runner = BenchmarkRunner(AlwaysSuccessAgent(), self._config(n_runs=1))
        results = runner.run()
        assert results[0].framework == "mock"
        assert results[0].task == "tool_use"

    def test_run_result_run_id_sequential(self):
        runner = BenchmarkRunner(AlwaysSuccessAgent(), self._config(n_runs=5))
        results = runner.run()
        assert [r.run_id for r in results] == [0, 1, 2, 3, 4]

    def test_run_result_latency_non_negative(self):
        runner = BenchmarkRunner(AlwaysSuccessAgent(), self._config(n_runs=3))
        for r in runner.run():
            assert r.latency_ms >= 0.0

    def test_failure_result_has_exception_type(self):
        runner = BenchmarkRunner(AlwaysFailAgent(), self._config(n_runs=3))
        for r in runner.run():
            assert not r.success
            assert r.exception_type == "RuntimeError"
            assert "Mock agent failure" in r.exception_msg

    def test_failure_result_has_category(self):
        runner = BenchmarkRunner(AlwaysFailAgent(), self._config(n_runs=3))
        for r in runner.run():
            assert r.failure_category != ""

    def test_failure_type_label_none_baseline(self):
        runner = BenchmarkRunner(AlwaysSuccessAgent(), self._config(n_runs=1))
        results = runner.run()
        assert results[0].failure_type == "none"

    def test_failure_type_label_from_config(self):
        fc = ToolFailureConfig(
            failure_type=ToolFailureType.TIMEOUT,
            failure_probability=0.0,  # don't actually inject
        )
        config = self._config(n_runs=1, failure_config=fc)
        runner = BenchmarkRunner(AlwaysSuccessAgent(), config)
        results = runner.run()
        assert results[0].failure_type == "timeout"

    def test_to_dict_has_required_keys(self):
        runner = BenchmarkRunner(AlwaysSuccessAgent(), self._config(n_runs=1))
        r = runner.run()[0]
        d = r.to_dict()
        for key in ["run_id", "framework", "task", "success", "latency_ms", "failure_type"]:
            assert key in d

    def test_custom_task_fn(self):
        inputs_seen = []

        def custom_task(run_id: int, task: str) -> dict:
            inputs_seen.append({"run_id": run_id, "task": task})
            return {"input": f"custom input {run_id}"}

        runner = BenchmarkRunner(
            AlwaysSuccessAgent(),
            self._config(n_runs=3),
            task_fn=custom_task,
        )
        runner.run()
        assert len(inputs_seen) == 3
        assert inputs_seen[0]["run_id"] == 0
        assert inputs_seen[2]["run_id"] == 2

    def test_save_results_creates_csv_and_json(self, tmp_path):
        save_path = str(tmp_path / "results" / "test_run")
        config = BenchmarkConfig(
            framework="mock",
            task="tool_use",
            n_runs=3,
            recovery_attempts=0,
            save_results_path=save_path,
        )
        runner = BenchmarkRunner(AlwaysSuccessAgent(), config)
        runner.run()

        import os
        assert os.path.exists(save_path + ".csv")
        assert os.path.exists(save_path + ".json")
