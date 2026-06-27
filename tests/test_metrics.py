"""Tests for MetricsCalculator — MTTR, failure rate, P99 latency."""

from __future__ import annotations

import pytest

from agentprobe.benchmark.metrics import AgentMetrics, MetricsCalculator
from agentprobe.benchmark.runner import RunResult


def _run(run_id: int, success: bool, latency: float = 100.0,
         recovered: bool = False, recovery_latency: float = 0.0) -> RunResult:
    return RunResult(
        run_id=run_id,
        framework="test",
        task="tool_use",
        failure_type="timeout",
        success=success,
        latency_ms=latency,
        recovered=recovered,
        recovery_latency_ms=recovery_latency,
    )


class TestMetricsCalculator:
    def test_empty_runs_returns_zero_metrics(self):
        m = MetricsCalculator().compute([])
        assert m.n_runs == 0
        assert m.failure_rate == 0.0
        assert m.p99_latency_ms == 0.0

    def test_all_success_zero_failure_rate(self):
        runs = [_run(i, success=True, latency=100.0) for i in range(10)]
        m = MetricsCalculator().compute(runs)
        assert m.failure_rate == 0.0
        assert m.n_successes == 10
        assert m.n_failures == 0

    def test_all_failure_rate_is_one(self):
        runs = [_run(i, success=False) for i in range(10)]
        m = MetricsCalculator().compute(runs)
        assert m.failure_rate == 1.0

    def test_mixed_failure_rate(self):
        runs = (
            [_run(i, success=True) for i in range(8)] +
            [_run(i + 8, success=False) for i in range(2)]
        )
        m = MetricsCalculator().compute(runs)
        assert abs(m.failure_rate - 0.2) < 0.001

    def test_recovery_rate_partial(self):
        runs = [_run(i, success=False, recovered=(i < 4)) for i in range(10)]
        m = MetricsCalculator().compute(runs)
        assert abs(m.recovery_rate - 0.4) < 0.001

    def test_recovery_rate_zero_when_no_failures(self):
        runs = [_run(i, success=True) for i in range(5)]
        m = MetricsCalculator().compute(runs)
        assert m.recovery_rate == 0.0

    def test_mttr_computed_correctly(self):
        runs = [_run(i, success=False, recovered=True, recovery_latency=1000.0) for i in range(5)]
        m = MetricsCalculator().compute(runs)
        assert m.mttr_ms == 1000.0

    def test_mttr_zero_when_no_recoveries(self):
        runs = [_run(i, success=False, recovered=False) for i in range(5)]
        m = MetricsCalculator().compute(runs)
        assert m.mttr_ms == 0.0

    def test_latency_percentiles_ordered(self):
        runs = [_run(i, success=True, latency=float(i)) for i in range(100)]
        m = MetricsCalculator().compute(runs)
        assert m.p50_latency_ms <= m.p90_latency_ms
        assert m.p90_latency_ms <= m.p99_latency_ms
        assert m.p99_latency_ms <= m.max_latency_ms

    def test_mean_latency_correct(self):
        runs = [_run(i, success=True, latency=100.0) for i in range(10)]
        m = MetricsCalculator().compute(runs)
        assert m.mean_latency_ms == 100.0

    def test_labels_inferred_from_runs(self):
        runs = [_run(0, success=True)]
        m = MetricsCalculator().compute(runs)
        assert m.framework == "test"
        assert m.task == "tool_use"
        assert m.failure_type == "timeout"

    def test_labels_overridden_by_explicit_args(self):
        runs = [_run(0, success=True)]
        m = MetricsCalculator().compute(runs, framework="langgraph", task="rag", failure_type="none")
        assert m.framework == "langgraph"
        assert m.task == "rag"
        assert m.failure_type == "none"

    def test_to_dict_contains_required_keys(self):
        runs = [_run(i, success=i % 2 == 0) for i in range(10)]
        m = MetricsCalculator().compute(runs)
        d = m.to_dict()
        required_keys = [
            "framework", "task", "failure_type", "n_runs",
            "failure_rate", "mttr_ms", "p50_latency_ms",
            "p90_latency_ms", "p99_latency_ms",
        ]
        for key in required_keys:
            assert key in d

    def test_compute_all_multi_config(self):
        configs = {
            "langgraph/tool_use/timeout": [_run(i, success=i % 3 != 0) for i in range(9)],
            "langchain/rag/none": [_run(i, success=True) for i in range(5)],
        }
        metrics_list = MetricsCalculator().compute_all(configs)
        assert len(metrics_list) == 2
        frameworks = {m.framework for m in metrics_list}
        assert "langgraph" in frameworks
        assert "langchain" in frameworks

    def test_str_representation(self):
        runs = [_run(i, success=i % 2 == 0, latency=200.0) for i in range(10)]
        m = MetricsCalculator().compute(runs)
        s = str(m)
        assert "failure_rate" in s
        assert "mttr" in s
        assert "p99" in s
