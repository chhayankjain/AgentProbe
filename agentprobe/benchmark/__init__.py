"""Benchmark suite for AgentProbe.

Usage::

    from agentprobe.benchmark import BenchmarkConfig, BenchmarkRunner, MetricsCalculator

    runner = BenchmarkRunner(agent=my_agent, config=BenchmarkConfig(n_runs=30))
    results = runner.run()
    metrics = MetricsCalculator().compute(results)
"""

from agentprobe.benchmark.grader import BatchGradeResult, GradeResult, ResponseGrader
from agentprobe.benchmark.metrics import AgentMetrics, MetricsCalculator
from agentprobe.benchmark.runner import (
    PAPER_PROMPTS,
    BenchmarkConfig,
    BenchmarkRunner,
    RunResult,
    paper_task_fn,
)
from agentprobe.benchmark.statistics import (
    FrameworkData,
    HypothesisTestResult,
    PairwiseComparison,
    StatisticalReport,
    build_comparison_report,
    cohens_d,
    cohens_h,
    compare_distributions,
    compare_multiple_groups,
    compare_proportions,
    mean_ci,
    proportion_ci,
)
from agentprobe.benchmark.tasks import corpus_task_fn, get_task

__all__ = [
    # Runner
    "BenchmarkConfig",
    "BenchmarkRunner",
    "RunResult",
    "PAPER_PROMPTS",
    "paper_task_fn",
    # Tasks
    "get_task",
    "corpus_task_fn",
    # Metrics
    "AgentMetrics",
    "MetricsCalculator",
    # Grading
    "ResponseGrader",
    "GradeResult",
    "BatchGradeResult",
    # Statistics
    "FrameworkData",
    "PairwiseComparison",
    "StatisticalReport",
    "HypothesisTestResult",
    "build_comparison_report",
    "cohens_d",
    "cohens_h",
    "compare_distributions",
    "compare_multiple_groups",
    "compare_proportions",
    "mean_ci",
    "proportion_ci",
]
