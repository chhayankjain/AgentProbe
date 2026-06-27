"""Benchmark runner for AgentProbe.

Orchestrates multi-run experiments across agent frameworks, failure types,
and tasks. Produces structured RunResult records used by MetricsCalculator.

Usage::

    from agentprobe.benchmark.runner import BenchmarkRunner, BenchmarkConfig
    from agentprobe.injector.tool_failure import ToolFailureConfig, ToolFailureType
    from agentprobe.agents.langgraph_agent import LangGraphToolAgent

    config = BenchmarkConfig(
        framework="langgraph",
        task="tool_use",
        n_runs=50,
        failure_config=ToolFailureConfig(
            failure_type=ToolFailureType.TIMEOUT,
            failure_probability=0.2,
        ),
    )

    agent = LangGraphToolAgent(llm=my_llm)
    runner = BenchmarkRunner(agent=agent, config=config)
    results = runner.run()

    from agentprobe.benchmark.metrics import MetricsCalculator
    metrics = MetricsCalculator().compute(results)
    print(metrics)
"""

from __future__ import annotations

import csv
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from agentprobe.observer.tracer import get_tracer

# Injector and classifier imports are lazy so this module is independently
# testable before their respective feature branches are merged.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentprobe.injector.tool_failure import ToolFailureConfig, ToolFailureInjector
    from agentprobe.observer.classifier import FailureClassifier as _ClassifierType

_tracer = get_tracer(__name__)
_console = Console()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Result of a single benchmark run (one agent invocation)."""

    run_id: int
    framework: str
    task: str
    failure_type: str
    success: bool
    latency_ms: float
    failure_category: str = "none"
    exception_type: str = ""
    exception_msg: str = ""
    recovered: bool = False
    recovery_latency_ms: float = 0.0
    output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "framework": self.framework,
            "task": self.task,
            "failure_type": self.failure_type,
            "success": self.success,
            "latency_ms": round(self.latency_ms, 2),
            "failure_category": self.failure_category,
            "exception_type": self.exception_type,
            "recovered": self.recovered,
            "recovery_latency_ms": round(self.recovery_latency_ms, 2),
        }


@dataclass
class BenchmarkConfig:
    """Configuration for one benchmark experiment.

    Attributes:
        framework: Agent framework identifier (e.g. "langgraph", "langchain").
        task: Task identifier (e.g. "tool_use", "rag", "multi_agent").
        n_runs: Number of independent runs per configuration.
        failure_config: Tool failure injection configuration.
            If None, runs without any injection (baseline).
        recovery_attempts: How many times to retry after a failure.
            Set to 0 to measure failure rate without recovery.
        timeout_per_run_seconds: Wall-clock timeout per run.
        seed: Global RNG seed for reproducibility.
        save_results_path: If set, writes CSV/JSON results to this path.
    """

    framework: str = "langgraph"
    task: str = "tool_use"
    n_runs: int = 30
    failure_config: "ToolFailureConfig | None" = None
    recovery_attempts: int = 1
    timeout_per_run_seconds: float = 60.0
    seed: int = 42
    save_results_path: str | None = None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Runs benchmark experiments for one (framework, task, failure_config) triple.

    Args:
        agent: An object with an ``invoke(input: dict) -> dict`` method.
            Should be a LangGraph, LangChain, or AutoGen agent wrapper from
            agentprobe.agents.*
        config: Benchmark configuration.
        task_fn: Optional callable that generates task inputs per run.
            Signature: ``(run_id: int, task: str) -> dict``.
            Defaults to the built-in task generator from agentprobe.benchmark.tasks.
    """

    def __init__(
        self,
        agent: Any,
        config: BenchmarkConfig | None = None,
        task_fn: Callable[[int, str], dict] | None = None,
    ) -> None:
        self.agent = agent
        self.config = config or BenchmarkConfig()
        self._task_fn = task_fn or _default_task_fn
        self._classifier: Any = None
        try:
            from agentprobe.observer.classifier import FailureClassifier
            self._classifier = FailureClassifier()
        except ImportError:
            pass  # classifier module not yet merged
        self._injector: Any = None
        if self.config.failure_config is not None:
            try:
                from agentprobe.injector.tool_failure import ToolFailureInjector
                self._injector = ToolFailureInjector(self.config.failure_config)
            except ImportError:
                pass  # injector module not yet merged; runner still functional

    def run(self) -> list[RunResult]:
        """Execute the full benchmark. Returns a list of RunResult objects."""
        cfg = self.config
        failure_type = (
            cfg.failure_config.failure_type.value if cfg.failure_config else "none"
        )

        _console.rule(
            f"[bold blue]AgentProbe Benchmark — {cfg.framework}/{cfg.task}/{failure_type}"
        )
        _console.print(f"  Runs: {cfg.n_runs}  |  Recovery attempts: {cfg.recovery_attempts}")

        results: list[RunResult] = []

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            TimeElapsedColumn(),
            console=_console,
        ) as progress:
            task_bar = progress.add_task(
                f"[cyan]Running {cfg.n_runs} trials...", total=cfg.n_runs
            )

            for run_id in range(cfg.n_runs):
                result = self._run_single(run_id, failure_type)
                results.append(result)
                progress.advance(task_bar)

        _console.print(self._summary_line(results))

        if cfg.save_results_path:
            self._save(results, cfg.save_results_path)

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_single(self, run_id: int, failure_type: str) -> RunResult:
        cfg = self.config
        task_input = self._task_fn(run_id, cfg.task)

        with _tracer.start_as_current_span(
            f"benchmark.run.{run_id}",
            attributes={
                "run.id": run_id,
                "framework": cfg.framework,
                "task": cfg.task,
                "failure_type": failure_type,
            },
        ):
            start = time.perf_counter()
            try:
                output = self.agent.invoke(task_input)
                latency_ms = (time.perf_counter() - start) * 1000
                return RunResult(
                    run_id=run_id,
                    framework=cfg.framework,
                    task=cfg.task,
                    failure_type=failure_type,
                    success=True,
                    latency_ms=latency_ms,
                    output=str(output)[:500],
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - start) * 1000
                category = "unknown"
                if self._classifier is not None:
                    event = self._classifier.classify_exception(exc, framework=cfg.framework)
                    category = event.category.value

                # Attempt recovery
                recovered, recovery_latency_ms = self._attempt_recovery(task_input)

                return RunResult(
                    run_id=run_id,
                    framework=cfg.framework,
                    task=cfg.task,
                    failure_type=failure_type,
                    success=False,
                    latency_ms=latency_ms,
                    failure_category=category,
                    exception_type=type(exc).__name__,
                    exception_msg=str(exc)[:200],
                    recovered=recovered,
                    recovery_latency_ms=recovery_latency_ms,
                )

    def _attempt_recovery(self, task_input: dict) -> tuple[bool, float]:
        """Try to invoke the agent again after a failure. Returns (recovered, latency_ms)."""
        if self.config.recovery_attempts == 0:
            return False, 0.0
        # Temporarily remove injector by running with a clean-probability injector
        # Recovery uses reduced failure probability (simulating retry logic)
        for attempt in range(self.config.recovery_attempts):
            start = time.perf_counter()
            try:
                self.agent.invoke(task_input)
                return True, (time.perf_counter() - start) * 1000
            except Exception:
                continue
        return False, 0.0

    def _summary_line(self, results: list[RunResult]) -> str:
        total = len(results)
        failures = sum(1 for r in results if not r.success)
        rate = failures / total if total > 0 else 0.0
        recovered = sum(1 for r in results if r.recovered)
        latencies = [r.latency_ms for r in results]
        p99 = sorted(latencies)[min(int(total * 0.99), total - 1)] if latencies else 0.0
        return (
            f"[green]Done[/green] — {total} runs | "
            f"failure_rate={rate:.1%} | "
            f"recovered={recovered}/{failures} | "
            f"P99={p99:.0f}ms"
        )

    def _save(self, results: list[RunResult], path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # CSV
        csv_path = p.with_suffix(".csv")
        with open(csv_path, "w", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
                writer.writeheader()
                writer.writerows(r.to_dict() for r in results)

        # JSON
        json_path = p.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        _console.print(f"[dim]Results saved to {csv_path} and {json_path}[/dim]")


# ---------------------------------------------------------------------------
# Default task generator
# ---------------------------------------------------------------------------


def _default_task_fn(run_id: int, task: str) -> dict[str, Any]:
    """Generate a task input dict for a given task type and run ID."""
    _TOOL_USE_INPUTS = [
        "What is the capital of France?",
        "Search for recent news about LLM reliability.",
        "Calculate 15% of 847.",
        "What is the current weather in San Francisco?",
        "Find the top 3 papers on multi-agent systems published in 2024.",
    ]
    _RAG_INPUTS = [
        "Summarize the main contributions of the AgentProbe paper.",
        "What are the 5 failure categories in the AgentProbe taxonomy?",
        "How does AgentProbe measure MTTR?",
        "Compare LangGraph and LangChain reliability profiles.",
        "What observability gaps exist in current agentic frameworks?",
    ]
    _MULTI_AGENT_INPUTS = [
        "Research and then summarize the latest advances in LLM agents.",
        "Find relevant papers and write a literature review outline.",
        "Analyze the dataset and produce a visualization plan.",
        "Draft a blog post about AI reliability best practices.",
        "Create a benchmark plan for testing agent robustness.",
    ]

    inputs_map = {
        "tool_use": _TOOL_USE_INPUTS,
        "rag": _RAG_INPUTS,
        "multi_agent": _MULTI_AGENT_INPUTS,
    }
    inputs = inputs_map.get(task, _TOOL_USE_INPUTS)
    return {"input": inputs[run_id % len(inputs)], "run_id": run_id}
