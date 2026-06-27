"""Reliability metrics for AgentProbe benchmarks.

Computes:
  - Failure rate        : fraction of runs that raised an exception
  - MTTR                : mean time to recovery (seconds from failure to next success)
  - P50 / P90 / P99     : latency percentiles across all runs
  - Recovery rate       : fraction of failures from which the agent recovered
  - Failure distribution: breakdown by failure category

All computations are pure Python + NumPy — no LLM calls.

Usage::

    from agentprobe.benchmark.metrics import MetricsCalculator
    from agentprobe.benchmark.runner import RunResult

    calc = MetricsCalculator()
    metrics = calc.compute(run_results)
    print(metrics.failure_rate)   # e.g. 0.12
    print(metrics.p99_latency_ms) # e.g. 4820.0

    df = metrics.to_dataframe()   # pandas DataFrame for paper plots
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# AgentMetrics dataclass
# ---------------------------------------------------------------------------


@dataclass
class AgentMetrics:
    """Aggregate reliability metrics for one (framework, task, failure_type) configuration.

    All latencies are in milliseconds.
    """

    framework: str
    task: str
    failure_type: str
    n_runs: int

    # Failure metrics
    failure_rate: float = 0.0           # [0, 1]
    n_failures: int = 0
    n_successes: int = 0

    # Recovery metrics
    recovery_rate: float = 0.0          # fraction of failures that recovered
    mttr_ms: float = 0.0               # mean time to recovery (ms)
    mttr_ms_std: float = 0.0

    # Latency percentiles (total turn latency including injected delays)
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # Failure breakdown by category
    failure_by_category: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "task": self.task,
            "failure_type": self.failure_type,
            "n_runs": self.n_runs,
            "failure_rate": round(self.failure_rate, 4),
            "n_failures": self.n_failures,
            "n_successes": self.n_successes,
            "recovery_rate": round(self.recovery_rate, 4),
            "mttr_ms": round(self.mttr_ms, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p90_latency_ms": round(self.p90_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
        }

    def to_dataframe_row(self) -> dict[str, Any]:
        return self.to_dict()

    def __str__(self) -> str:
        return (
            f"AgentMetrics({self.framework}/{self.task}/{self.failure_type}) "
            f"failure_rate={self.failure_rate:.1%} "
            f"mttr={self.mttr_ms:.0f}ms "
            f"p99={self.p99_latency_ms:.0f}ms"
        )


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


class MetricsCalculator:
    """Computes AgentProbe reliability metrics from a list of RunResult objects."""

    def compute(self, runs: list[Any], framework: str = "", task: str = "", failure_type: str = "") -> AgentMetrics:
        """Compute metrics from a list of :class:`~agentprobe.benchmark.runner.RunResult`.

        Args:
            runs: List of RunResult objects.
            framework: Framework name (used for labeling; inferred from runs if empty).
            task: Task name (used for labeling; inferred from runs if empty).
            failure_type: Failure type (used for labeling; inferred from runs if empty).
        """
        if not runs:
            return AgentMetrics(
                framework=framework, task=task, failure_type=failure_type, n_runs=0
            )

        # Infer labels from first run if not provided
        first = runs[0]
        fw = framework or getattr(first, "framework", "unknown")
        tk = task or getattr(first, "task", "unknown")
        ft = failure_type or getattr(first, "failure_type", "none")

        successes = [r for r in runs if getattr(r, "success", False)]
        failures = [r for r in runs if not getattr(r, "success", False)]
        recovered = [r for r in failures if getattr(r, "recovered", False)]

        latencies = [getattr(r, "latency_ms", 0.0) for r in runs]
        recovery_latencies = [getattr(r, "recovery_latency_ms", 0.0) for r in recovered]

        n = len(runs)
        failure_rate = len(failures) / n
        recovery_rate = len(recovered) / len(failures) if failures else 0.0

        return AgentMetrics(
            framework=fw,
            task=tk,
            failure_type=ft,
            n_runs=n,
            failure_rate=failure_rate,
            n_failures=len(failures),
            n_successes=len(successes),
            recovery_rate=recovery_rate,
            mttr_ms=statistics.mean(recovery_latencies) if recovery_latencies else 0.0,
            mttr_ms_std=statistics.stdev(recovery_latencies) if len(recovery_latencies) > 1 else 0.0,
            mean_latency_ms=statistics.mean(latencies) if latencies else 0.0,
            std_latency_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            p50_latency_ms=_percentile(latencies, 50),
            p90_latency_ms=_percentile(latencies, 90),
            p99_latency_ms=_percentile(latencies, 99),
            max_latency_ms=max(latencies) if latencies else 0.0,
            failure_by_category=self._count_by_category(failures),
        )

    def compute_all(self, results_by_config: dict[str, list[Any]]) -> list[AgentMetrics]:
        """Compute metrics for multiple configurations.

        Args:
            results_by_config: Maps config label (e.g. "langgraph/rag/timeout")
                to a list of RunResult objects.
        """
        metrics = []
        for config_label, runs in results_by_config.items():
            parts = config_label.split("/")
            fw = parts[0] if len(parts) > 0 else ""
            tk = parts[1] if len(parts) > 1 else ""
            ft = parts[2] if len(parts) > 2 else ""
            metrics.append(self.compute(runs, framework=fw, task=tk, failure_type=ft))
        return metrics

    def to_dataframe(self, metrics_list: list[AgentMetrics]) -> Any:
        """Convert a list of AgentMetrics to a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for to_dataframe(). pip install pandas") from e
        rows = [m.to_dict() for m in metrics_list]
        return pd.DataFrame(rows)

    def _count_by_category(self, failures: list[Any]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in failures:
            cat = getattr(r, "failure_category", "unknown")
            counts[cat] = counts.get(cat, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile(data: list[float], pct: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = max(0, int(len(sorted_data) * pct / 100) - 1)
    return sorted_data[idx]
