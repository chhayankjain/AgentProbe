"""Pydantic models and enums for the AgentProbe benchmark subsystem.

Usage::

    from agentprobe.benchmark.types import BenchmarkConfig, TaskType, AgentFramework
    from agentprobe.injector.types import ToolFailureConfig, ToolFailureType

    config = BenchmarkConfig(
        framework=AgentFramework.LANGGRAPH,
        task=TaskType.TOOL_USE,
        n_runs=50,
        failure_config=ToolFailureConfig(
            failure_type=ToolFailureType.TIMEOUT,
            failure_probability=0.2,
            seed=42,
        ),
        save_results_path="experiments/results/run_01",
    )
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from agentprobe.injector.types import ToolFailureConfig, ToolFailureType


class TaskType(str, Enum):
    """Benchmark task category."""

    TOOL_USE = "tool_use"
    RAG = "rag"
    MULTI_AGENT = "multi_agent"


class AgentFramework(str, Enum):
    """Supported agent framework backends."""

    LANGGRAPH = "langgraph"
    LANGCHAIN = "langchain"
    AUTOGEN = "autogen"


class RunResult(BaseModel):
    """Outcome of a single benchmark run.

    Args:
        run_id: Unique identifier for this run (e.g. UUID or sequential index).
        framework: Agent framework used.
        task: Benchmark task category.
        success: Whether the agent completed the task successfully.
        failure_type: The failure mode that was active during this run.
        duration_ms: Wall-clock time from task start to task end in milliseconds.
        recovery_time_ms: Time from first failure to successful recovery.
            ``None`` if no failure occurred or recovery never succeeded.
        retries: Number of retry attempts made by the agent.
        error_message: Human-readable error detail when ``success`` is ``False``.
        timestamp: UTC datetime when the run started.
    """

    run_id: str
    framework: AgentFramework
    task: TaskType
    success: bool
    failure_type: ToolFailureType
    duration_ms: float
    recovery_time_ms: float | None = None
    retries: int = 0
    error_message: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BenchmarkConfig(BaseModel):
    """Full configuration for a benchmark experiment.

    Args:
        framework: Agent framework to evaluate.
        task: Task category to run.
        n_runs: Number of independent runs to execute.
        failure_config: Failure injection settings.
        recovery_attempts: Maximum retries the agent may make per run.
        save_results_path: Directory path for writing ``BenchmarkResults`` JSON.
            ``None`` disables file output.
    """

    framework: AgentFramework
    task: TaskType
    n_runs: int = Field(default=10, ge=1)
    failure_config: ToolFailureConfig = Field(default_factory=ToolFailureConfig)
    recovery_attempts: int = Field(default=3, ge=0)
    save_results_path: str | None = None


class BenchmarkResults(BaseModel):
    """All run results for a single benchmark experiment.

    Args:
        config: The configuration that produced these results.
        runs: Individual :class:`RunResult` records.
        total_duration_ms: Wall-clock time for the entire experiment.
    """

    config: BenchmarkConfig
    runs: list[RunResult]
    total_duration_ms: float


class BenchmarkMetrics(BaseModel):
    """Computed reliability metrics derived from :class:`BenchmarkResults`.

    Args:
        failure_rate: Fraction of runs that failed (0.0–1.0).
        mttr_ms: Mean Time To Recovery across all runs with a recorded recovery.
            ``0.0`` when no failures were injected or no recovery was observed.
        p99_latency_ms: 99th-percentile run duration in milliseconds.
        p50_latency_ms: Median run duration in milliseconds.
        total_runs: Total number of runs in the experiment.
        successful_runs: Number of runs where ``success`` is ``True``.
    """

    failure_rate: float
    mttr_ms: float
    p99_latency_ms: float
    p50_latency_ms: float
    total_runs: int
    successful_runs: int
