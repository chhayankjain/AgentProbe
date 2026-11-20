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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opentelemetry import context as otel_context
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from agentprobe._threads import start_daemon
from agentprobe.benchmark.metrics import _percentile
from agentprobe.errors import HarnessLimitation
from agentprobe.llm import is_llm_backend_error
from agentprobe.observer.classifier import FailureClassifier
from agentprobe.observer.tracer import get_tracer

if TYPE_CHECKING:
    from agentprobe.injector.tool_failure import ToolFailureConfig

_tracer = get_tracer(__name__)
_console = Console()


def _is_harness_limitation(exc: BaseException) -> bool:
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, HarnessLimitation):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


class _HarnessTimeout(Exception):
    """Raised when the benchmark runner's wall-clock deadline is exceeded.

    Distinct from ``TimeoutError`` so that upstream timeouts (e.g. pyautogen's
    ``Completion.request_timeout``) are not misclassified as harness deadlines.
    """


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Result of a single benchmark run (one agent invocation).

    The ``status`` field distinguishes executed trials from those that
    produced no valid observation:
      - ``"completed"``     — trial was dispatched and finished (success or failure).
      - ``"skipped"``       — trial was never dispatched (prior timeout lingering).
      - ``"backend_error"`` — an LLM model call failed at the provider or its
        transport (429, 5xx, connection error) even after in-place retries.
        This is an infrastructure event, not an agent failure, and is kept
        separate from *injected* tool faults. Identified at the LLM-call
        boundary, so a tool that fails with the same transport error is an
        ordinary failure.
      - ``"harness_error"`` — the adapter could not run the trial faithfully
        (e.g. parallel tool calls the AutoGen adapter cannot execute).

    Tool evidence is per trial: ``trial_tool_calls``/``trial_injections`` count
    tool calls and injected faults during the initial invocation (summed over
    any backend re-dispatches), ``recovery_tool_calls``/``recovery_injections``
    during the recovery attempt.

    Only ``"completed"`` trials should be included in error-rate, recovery,
    and latency calculations.
    """

    run_id: int
    framework: str
    task: str
    failure_type: str
    success: bool
    latency_ms: float
    status: str = "completed"
    failure_category: str = "none"
    exception_type: str = ""
    exception_msg: str = ""
    recovered: bool = False
    recovery_latency_ms: float = 0.0
    recovery_output: str = ""
    #: "recovered" | "failed" | "unassessed" (the recovery call hit an LLM backend
    #: error even after retries, so nothing can be said about recovery) |
    #: "not_attempted" (recovery_attempts == 0) | "n/a" (the run succeeded).
    recovery_status: str = "n/a"
    output: str = ""
    backend_retries: int = 0
    trial_tool_calls: int = 0
    trial_injections: int = 0
    recovery_tool_calls: int = 0
    recovery_injections: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "framework": self.framework,
            "task": self.task,
            "failure_type": self.failure_type,
            "status": self.status,
            "success": self.success,
            "latency_ms": round(self.latency_ms, 2),
            "failure_category": self.failure_category,
            "exception_type": self.exception_type,
            "exception_msg": self.exception_msg,
            "recovered": self.recovered,
            "recovery_latency_ms": round(self.recovery_latency_ms, 2),
            "recovery_output": self.recovery_output,
            "recovery_status": self.recovery_status,
            "output": self.output,
            "backend_retries": self.backend_retries,
            "trial_tool_calls": self.trial_tool_calls,
            "trial_injections": self.trial_injections,
            "recovery_tool_calls": self.recovery_tool_calls,
            "recovery_injections": self.recovery_injections,
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
        backend_error_retries: How many times to re-dispatch a trial whose
            *LLM provider* failed (429/5xx/connection error) before recording
            it as ``status="backend_error"``. Injected tool faults never
            trigger this path.
        backend_error_backoff_seconds: Sleep before each backend-error
            re-dispatch (doubles each attempt).
    """

    framework: str = "langgraph"
    task: str = "tool_use"
    n_runs: int = 30
    failure_config: ToolFailureConfig | None = None
    recovery_attempts: int = 1
    timeout_per_run_seconds: float = 60.0
    seed: int = 42
    save_results_path: str | None = None
    backend_error_retries: int = 2
    backend_error_backoff_seconds: float = 30.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Runs benchmark experiments for one (framework, task, failure_config) triple.

    Args:
        agent: Any object satisfying :class:`agentprobe.agents.BaseAgent`
            (``invoke(input: dict) -> dict`` plus ``reset()``), e.g. the
            LangGraph, LangChain, or AutoGen adapters in ``agentprobe.agents``.
        config: Benchmark configuration.
        task_fn: Optional callable that generates task inputs per run.
            Signature: ``(run_id: int, task: str) -> dict``.
            Defaults to :func:`paper_task_fn`, the five-prompt set used in the
            paper. Use ``agentprobe.benchmark.tasks.corpus_task_fn(task)`` to
            run against the larger graded corpora instead.
    """

    def __init__(
        self,
        agent: Any,
        config: BenchmarkConfig | None = None,
        task_fn: Callable[[int, str], dict[str, Any]] | None = None,
    ) -> None:
        self.agent = agent
        self.config = config or BenchmarkConfig()
        self._task_fn = task_fn or paper_task_fn
        self._classifier: Any = FailureClassifier()
        self._invocation_hung = False

    def run(self) -> list[RunResult]:
        """Execute the full benchmark. Returns a list of RunResult objects."""
        cfg = self.config
        failure_type = cfg.failure_config.failure_type.value if cfg.failure_config else "none"

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
            task_bar = progress.add_task(f"[cyan]Running {cfg.n_runs} trials...", total=cfg.n_runs)

            for run_id in range(cfg.n_runs):
                if self._invocation_hung:
                    results.append(
                        RunResult(
                            run_id=run_id,
                            framework=cfg.framework,
                            task=cfg.task,
                            failure_type=failure_type,
                            success=False,
                            latency_ms=0.0,
                            status="skipped",
                            failure_category="harness_abort",
                            exception_type="_HarnessTimeout",
                            exception_msg="Skipped: prior trial timed out with lingering thread",
                        )
                    )
                    progress.advance(task_bar)
                    continue
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
        """Run one trial, re-dispatching on LLM backend errors.

        A provider-side failure (rate limit, 5xx, connection reset) is not an
        observation about the agent, so the trial is retried in place up to
        ``backend_error_retries`` times with exponential backoff. If it still
        fails, the trial is recorded with ``status="backend_error"`` and
        excluded from all metrics.
        """
        cfg = self.config
        attempt = 0
        prior_calls = prior_injections = 0  # tool activity in abandoned dispatches
        while True:
            result = self._run_single_once(run_id, failure_type)
            result.trial_tool_calls += prior_calls
            result.trial_injections += prior_injections
            if result.status != "backend_error":
                result.backend_retries += attempt
                return result
            if attempt >= cfg.backend_error_retries or self._invocation_hung:
                result.backend_retries += attempt
                return result
            prior_calls, prior_injections = result.trial_tool_calls, result.trial_injections
            delay = cfg.backend_error_backoff_seconds * (2**attempt)
            _console.print(
                f"  [yellow]LLM backend error on run {run_id} "
                f"({result.exception_type}); retrying in {delay:.0f}s "
                f"({attempt + 1}/{cfg.backend_error_retries})[/yellow]"
            )
            if delay > 0:
                time.sleep(delay)
            attempt += 1

    def _run_single_once(self, run_id: int, failure_type: str) -> RunResult:
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
            c0 = self._tool_counts()
            try:
                output = self._invoke_with_timeout(task_input)
                latency_ms = (time.perf_counter() - start) * 1000
                calls, injections = self._delta(c0)
                return RunResult(
                    run_id=run_id,
                    framework=cfg.framework,
                    task=cfg.task,
                    failure_type=failure_type,
                    success=True,
                    latency_ms=latency_ms,
                    output=str(output),
                    trial_tool_calls=calls,
                    trial_injections=injections,
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - start) * 1000
                calls, injections = self._delta(c0)
                if _is_harness_limitation(exc):
                    return RunResult(
                        run_id=run_id,
                        framework=cfg.framework,
                        task=cfg.task,
                        failure_type=failure_type,
                        success=False,
                        latency_ms=latency_ms,
                        status="harness_error",
                        failure_category="harness",
                        exception_type=type(exc).__name__,
                        exception_msg=str(exc)[:200],
                        trial_tool_calls=calls,
                        trial_injections=injections,
                    )
                if not isinstance(exc, _HarnessTimeout) and is_llm_backend_error(exc):
                    # Infrastructure failure: no recovery attempt, not an agent failure.
                    return RunResult(
                        run_id=run_id,
                        framework=cfg.framework,
                        task=cfg.task,
                        failure_type=failure_type,
                        success=False,
                        latency_ms=latency_ms,
                        status="backend_error",
                        failure_category="infrastructure",
                        exception_type=type(exc).__name__,
                        exception_msg=str(exc)[:200],
                        trial_tool_calls=calls,
                        trial_injections=injections,
                    )
                category = "unknown"
                if self._classifier is not None:
                    event = self._classifier.classify_exception(exc, framework=cfg.framework)
                    category = event.category.value
                    if category == "infrastructure":
                        # Provider-like class, but not raised by a model call (e.g. a
                        # real tool's httpx timeout): the origin is not established.
                        category = "unknown"

                # Attempt recovery
                rc0 = self._tool_counts()
                (
                    recovered,
                    recovery_latency_ms,
                    recovery_output,
                    recovery_status,
                    recovery_backend_retries,
                ) = self._attempt_recovery(task_input)
                rec_calls, rec_injections = self._delta(rc0)

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
                    recovery_output=str(recovery_output) if recovery_output else "",
                    recovery_status=recovery_status,
                    backend_retries=recovery_backend_retries,
                    trial_tool_calls=calls,
                    trial_injections=injections,
                    recovery_tool_calls=rec_calls,
                    recovery_injections=rec_injections,
                )

    def _invoke_with_timeout(self, task_input: dict[str, Any]) -> Any:
        """Invoke agent with wall-clock timeout enforcement.

        Uses a thread so that the agent's injector RNG, injection logs,
        and all mutable state advance in-process (no fork/copy issues).
        The current OpenTelemetry context is explicitly propagated into the
        worker thread so that agent spans become children of the benchmark span.

        Guarantees:
        1. Upstream exceptions (including ``TimeoutError``) are propagated
           with their original type and message.
        2. Only genuine wall-clock deadline expiry raises ``_HarnessTimeout``.
        3. After a harness timeout, ``self._invocation_hung`` is set to
           ``True`` so that ``_attempt_recovery`` skips retry and
           ``run()`` aborts remaining trials, preventing overlap between
           the lingering thread and new invocations.

        Limitation: Python threads cannot be forcibly killed.  A hung
        thread may linger until the process exits.
        """
        timeout = self.config.timeout_per_run_seconds
        ctx = otel_context.get_current()

        def _worker() -> Any:
            token = otel_context.attach(ctx)
            try:
                return self.agent.invoke(task_input)
            finally:
                otel_context.detach(token)

        future = start_daemon(_worker)

        try:
            return future.result(timeout=timeout)
        except Exception:
            if future.done():
                # Worker finished (possibly with an exception) before
                # the deadline — propagate the original exception as-is.
                return future.result()  # re-raises the worker exception
            # Worker still running — genuine deadline miss.
            self._invocation_hung = True
            raise _HarnessTimeout(f"Agent invocation exceeded {timeout}s wall-clock timeout")

    def _attempt_recovery(self, task_input: dict[str, Any]) -> tuple[bool, float, Any, str, int]:
        """Try to invoke the agent again after a failure.

        Returns ``(recovered, latency_ms, output, recovery_status, backend_retries)``.

        * An agent failure during the retry consumes one recovery attempt and,
          once attempts are exhausted, yields ``"failed"``.
        * An LLM *backend* error (429/5xx/connection) during the retry does not
          consume a recovery attempt: it is retried with backoff up to
          ``backend_error_retries`` and, if it persists, yields ``"unassessed"``
          so the trial is excluded from the recovery-rate denominator rather
          than counted as the agent failing to recover.
        * If the prior invocation is still running (hung thread after a harness
          timeout), recovery is skipped entirely (``"unassessed"``) to prevent
          two concurrent invocations on the same agent.
        """
        cfg = self.config
        if cfg.recovery_attempts == 0:
            return False, 0.0, None, "not_attempted", 0
        if self._invocation_hung:
            return False, 0.0, None, "unassessed", 0

        backend_retries = 0
        attempts_left = cfg.recovery_attempts
        while attempts_left > 0:
            start = time.perf_counter()
            try:
                output = self._invoke_with_timeout(task_input)
                return (
                    True,
                    (time.perf_counter() - start) * 1000,
                    output,
                    "recovered",
                    backend_retries,
                )
            except Exception as exc:
                if self._invocation_hung or _is_harness_limitation(exc):
                    return False, 0.0, None, "unassessed", backend_retries
                if not isinstance(exc, _HarnessTimeout) and is_llm_backend_error(exc):
                    if backend_retries >= cfg.backend_error_retries:
                        return False, 0.0, None, "unassessed", backend_retries
                    delay = cfg.backend_error_backoff_seconds * (2**backend_retries)
                    _console.print(
                        f"  [yellow]LLM backend error during recovery "
                        f"({type(exc).__name__}); retrying in {delay:.0f}s[/yellow]"
                    )
                    if delay > 0:
                        time.sleep(delay)
                    backend_retries += 1
                    continue  # a backend error does not consume a recovery attempt
                attempts_left -= 1
        return False, 0.0, None, "failed", backend_retries

    def _tool_counts(self) -> tuple[int, int]:
        injector = getattr(self.agent, "injector", None)
        if injector is None or not hasattr(injector, "counts"):
            return (0, 0)
        return injector.counts()  # type: ignore[no-any-return]

    def _delta(self, before: tuple[int, int]) -> tuple[int, int]:
        after = self._tool_counts()
        return after[0] - before[0], after[1] - before[1]

    def _summary_line(self, results: list[RunResult]) -> str:
        executed = [r for r in results if r.status == "completed"]
        backend_errors = sum(1 for r in results if r.status == "backend_error")
        harness_errors = sum(1 for r in results if r.status == "harness_error")
        skipped = len(results) - len(executed) - backend_errors - harness_errors
        total = len(executed)
        failures = sum(1 for r in executed if not r.success)
        unassessed = sum(1 for r in executed if not r.success and r.recovery_status == "unassessed")
        rate = failures / total if total > 0 else 0.0
        recovered = sum(1 for r in executed if r.recovered)
        latencies = [r.latency_ms for r in executed]
        p99 = _percentile(latencies, 99)
        rate_text = f"{rate:.1%}" if total else "N/A"
        assessable = failures - unassessed
        recovery_text = f"{recovered}/{assessable} assessable" if assessable else "N/A"
        line = (
            f"[green]Done[/green] — {total} executed | "
            f"failure_rate={rate_text} | "
            f"recovered={recovery_text} | "
            f"P99={p99:.0f}ms"
        )
        if skipped:
            line += f" | [yellow]{skipped} skipped (condition incomplete)[/yellow]"
        if backend_errors:
            line += f" | [red]{backend_errors} LLM backend errors (excluded)[/red]"
        if harness_errors:
            line += f" | [red]{harness_errors} harness limitations (excluded)[/red]"
        if unassessed:
            line += f" | [yellow]{unassessed} recoveries unassessed[/yellow]"
        return line

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
# Default task generator — the paper's prompt set
# ---------------------------------------------------------------------------

#: The five prompts per task type used in every experiment reported in the paper
#: (Phase 1, Phase 2, and cross-model validation). Run ``i`` receives prompt
#: ``i % 5``. Kept verbatim so that published batches reproduce exactly.
#: The larger graded corpora in :mod:`agentprobe.benchmark.tasks` are *not* the
#: paper's prompts; use ``corpus_task_fn()`` from that module to benchmark
#: against them instead.
PAPER_PROMPTS: dict[str, list[str]] = {
    "tool_use": [
        "What is the capital of France?",
        "Search for recent news about LLM reliability.",
        "Calculate 15% of 847.",
        "What is the current weather in San Francisco?",
        "Find the top 3 papers on multi-agent systems published in 2024.",
    ],
    "rag": [
        "Summarize the main contributions of the AgentProbe paper.",
        "What are the 5 failure categories in the AgentProbe taxonomy?",
        "How does AgentProbe measure MTTR?",
        "Compare LangGraph and LangChain reliability profiles.",
        "What observability gaps exist in current agentic frameworks?",
    ],
    "multi_agent": [
        "Research and then summarize the latest advances in LLM agents.",
        "Find relevant papers and write a literature review outline.",
        "Analyze the dataset and produce a visualization plan.",
        "Draft a blog post about AI reliability best practices.",
        "Create a benchmark plan for testing agent robustness.",
    ],
}


def paper_task_fn(run_id: int, task: str) -> dict[str, Any]:
    """Default ``task_fn``: cycle through :data:`PAPER_PROMPTS` for *task*.

    Unknown task names fall back to the ``tool_use`` prompts.
    """
    inputs = PAPER_PROMPTS.get(task, PAPER_PROMPTS["tool_use"])
    return {"input": inputs[run_id % len(inputs)], "run_id": run_id}


# Backwards-compatible alias used by the experiment scripts.
_default_task_fn = paper_task_fn
