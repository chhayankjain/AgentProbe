"""AgentProbe experiment runner v2 — addresses publication review concerns.

Improvements over v1:
  - Per-condition checkpointing (resume after interruption)
  - Correctness grading via ResponseGrader
  - Tool-call evidence from injector logs
  - Full input/output recording (no truncation)
  - Matched injection probability across all fault types
  - Separate output directory per experiment batch
  - Immutable experiment manifest and metadata
  - Validated condition completion before checkpointing
  - Non-zero exit on incomplete/failed batches
  - CLI --seed propagates to all failure configs

Usage::

    # Pilot (2 runs per condition, single framework)
    .venv/bin/python experiments/run_experiment_v2.py --pilot

    # Full matrix
    .venv/bin/python experiments/run_experiment_v2.py --n-runs 30

    # Resume interrupted experiment
    .venv/bin/python experiments/run_experiment_v2.py --resume experiments/results_v2/20261110_143022
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from agentprobe.benchmark.grader import ResponseGrader
from agentprobe.benchmark.metrics import MetricsCalculator
from agentprobe.benchmark.runner import BenchmarkConfig, BenchmarkRunner, RunResult
from agentprobe.benchmark.metadata import capture_metadata
from agentprobe.injector.tool_failure import ToolFailureConfig, ToolFailureType

_console = Console()

# ---------------------------------------------------------------------------
# Task definitions with correctness criteria
# ---------------------------------------------------------------------------

# Expected keywords for grading each prompt (index matches _default_task_fn)
TASK_GRADING = {
    "tool_use": [
        # "What is the capital of France?"
        {"keywords": ["paris"], "exact_answer": None},
        # "Search for recent news about LLM reliability."
        {"keywords": ["search", "results", "llm"], "exact_answer": None},
        # "Calculate 15% of 847."
        {"keywords": ["127.05"], "exact_answer": "127.05"},
        # "What is the current weather in San Francisco?"
        {"keywords": ["search", "results", "san francisco"], "exact_answer": None},
        # "Find the top 3 papers on multi-agent systems published in 2024."
        {"keywords": ["search", "results", "papers"], "exact_answer": None},
    ],
    "rag": [
        {"keywords": ["agentprobe"], "exact_answer": None},
        {"keywords": ["failure", "categor"], "exact_answer": None},
        {"keywords": ["recovery", "time"], "exact_answer": None},
        {"keywords": ["langgraph", "langchain"], "exact_answer": None},
        {"keywords": ["observ", "gap"], "exact_answer": None},
    ],
    "multi_agent": [
        {"keywords": ["agent", "llm"], "exact_answer": None},
        {"keywords": ["paper", "review"], "exact_answer": None},
        {"keywords": ["data", "visual"], "exact_answer": None},
        {"keywords": ["blog", "ai"], "exact_answer": None},
        {"keywords": ["benchmark", "test"], "exact_answer": None},
    ],
}

# Matched injection probability for all fault types (reviewer concern #7)
INJECTION_PROBABILITY = 0.2

FAILURE_TYPE_NAMES = ["none", "timeout", "malformed_output", "rate_limit", "api_error"]


def build_failure_configs(seed: int) -> dict[str, ToolFailureConfig | None]:
    """Construct failure configs with the given seed.

    All fault types share the same probability and seed so that injection
    schedules are comparable and the CLI --seed actually controls the RNG.
    """
    return {
        "none": None,
        "timeout": ToolFailureConfig(
            failure_type=ToolFailureType.TIMEOUT,
            failure_probability=INJECTION_PROBABILITY,
            timeout_seconds=5.0,
            seed=seed,
        ),
        "malformed_output": ToolFailureConfig(
            failure_type=ToolFailureType.MALFORMED_OUTPUT,
            failure_probability=INJECTION_PROBABILITY,
            seed=seed,
        ),
        "rate_limit": ToolFailureConfig(
            failure_type=ToolFailureType.RATE_LIMIT,
            failure_probability=INJECTION_PROBABILITY,
            retry_after_seconds=1.0,
            seed=seed,
        ),
        "api_error": ToolFailureConfig(
            failure_type=ToolFailureType.API_ERROR,
            failure_probability=INJECTION_PROBABILITY,
            error_code=503,
            seed=seed,
        ),
    }


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_llm(model: str) -> Any:
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, temperature=0.0)


def build_agent(framework: str, llm: Any, failure_config: ToolFailureConfig | None,
                model: str = "llama3.1") -> Any:
    if framework == "langgraph":
        from agentprobe.agents.langgraph_agent import LangGraphToolAgent
        return LangGraphToolAgent(llm=llm, failure_config=failure_config)
    elif framework == "langchain":
        from agentprobe.agents.langchain_agent import LangChainToolAgent
        return LangChainToolAgent(llm=llm, failure_config=failure_config)
    elif framework == "autogen":
        from agentprobe.agents.autogen_agent import AutoGenToolAgent
        llm_config = {
            "config_list": [{
                "model": model,
                "api_base": "http://localhost:11434/v1",
                "api_key": "ollama",
            }]
        }
        return AutoGenToolAgent(llm_config=llm_config, failure_config=failure_config)
    else:
        raise ValueError(f"Unknown framework: {framework}")


# ---------------------------------------------------------------------------
# Enhanced run record
# ---------------------------------------------------------------------------

@dataclass
class EnhancedRunRecord:
    """Complete record for one trial, addressing reviewer requirements."""
    # Identity
    run_id: int
    framework: str
    task: str
    failure_type: str
    batch_id: str

    # Execution
    status: str  # "completed" or "skipped"
    success: bool
    latency_ms: float
    output: str  # full, no truncation
    exception_type: str = ""
    exception_msg: str = ""  # full, no truncation

    # Recovery
    recovered: bool = False
    recovery_latency_ms: float = 0.0
    recovery_output: str = ""

    # Tool evidence (reviewer concern #3, #4)
    tool_call_count: int = 0
    injector_calls: int = 0
    injector_injections: int = 0

    # Correctness (reviewer concern #4.2)
    grade_score: float = 0.0
    grade_correct: bool = False
    grade_keywords_matched: str = ""
    expected_answer: str = ""

    # Input identity (reviewer concern #5)
    input_text: str = ""
    prompt_index: int = 0

    # Classification
    failure_category: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# Experiment state (checkpoint)
# ---------------------------------------------------------------------------

@dataclass
class ExperimentState:
    """Tracks which conditions have been completed for resume support.

    Stores both a flat key (for file naming / backward compat) and
    structured triples (framework, task, failure_type) so validation
    can check record identity without ambiguous underscore splitting.
    """
    completed_conditions: list[str] = field(default_factory=list)
    # condition key = "framework_task_failuretype"
    # Structured map: key -> (framework, task, failure_type)
    _condition_triples: dict[str, tuple[str, str, str]] = field(
        default_factory=dict, repr=False
    )

    def _key(self, framework: str, task: str, failure_type: str) -> str:
        return f"{framework}_{task}_{failure_type}"

    def is_done(self, framework: str, task: str, failure_type: str) -> bool:
        return self._key(framework, task, failure_type) in self.completed_conditions

    def mark_done(self, framework: str, task: str, failure_type: str) -> None:
        key = self._key(framework, task, failure_type)
        if key not in self.completed_conditions:
            self.completed_conditions.append(key)
        self._condition_triples[key] = (framework, task, failure_type)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump({
                "completed_conditions": self.completed_conditions,
                "condition_triples": {
                    k: list(v) for k, v in self._condition_triples.items()
                },
            }, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ExperimentState":
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            state = cls(
                completed_conditions=data.get("completed_conditions", [])
            )
            # Restore structured triples if present (new format)
            for k, v in data.get("condition_triples", {}).items():
                if isinstance(v, list) and len(v) == 3:
                    state._condition_triples[k] = tuple(v)
            return state
        return cls()

    def validate_checkpoint(
        self, batch_dir: Path, n_runs: int
    ) -> list[str]:
        """Validate every checkpointed condition:
        - Results file exists
        - Exactly n_runs completed records (not fewer, not more)
        - No duplicate run IDs
        - Every record's framework/task/failure_type matches the expected triple

        Returns a list of error messages (empty = all valid).
        """
        errors: list[str] = []
        for key in self.completed_conditions:
            triple = self._condition_triples.get(key)
            json_path = batch_dir / f"{key}.json"

            if not json_path.exists():
                errors.append(
                    f"Checkpoint claims '{key}' is done but "
                    f"{json_path.name} is missing"
                )
                continue

            with open(json_path) as f:
                records = json.load(f)
            completed = [r for r in records if r.get("status") == "completed"]

            # Exact count
            if len(completed) != n_runs:
                errors.append(
                    f"{json_path.name}: expected {n_runs} completed "
                    f"records, found {len(completed)}"
                )
                continue

            # Exact expected run IDs: 0 .. n_runs-1
            expected_ids = set(range(n_runs))
            run_ids = [r.get("run_id") for r in completed]
            actual_ids = set(run_ids)
            if len(actual_ids) < len(run_ids):
                n_dupes = len(run_ids) - len(actual_ids)
                errors.append(
                    f"{json_path.name}: {n_dupes} duplicate run IDs"
                )
            if actual_ids != expected_ids:
                missing = expected_ids - actual_ids
                extra = actual_ids - expected_ids
                parts = []
                if missing:
                    parts.append(f"missing IDs {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
                if extra:
                    parts.append(f"unexpected IDs {sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}")
                errors.append(
                    f"{json_path.name}: run ID mismatch — {', '.join(parts)}"
                )

            # Identity match — require every record's condition labels
            # to match the expected triple. For legacy checkpoints
            # without stored triples, reject with a migration message.
            if triple:
                exp_fw, exp_task, exp_ft = triple
            else:
                errors.append(
                    f"Legacy checkpoint without structured triples for "
                    f"'{key}'. Re-run or delete the checkpoint to "
                    f"enable full validation."
                )
                continue

            for r in completed:
                r_fw = r.get("framework", "")
                r_task = r.get("task", "")
                r_ft = r.get("failure_type", "")
                if (r_fw, r_task, r_ft) != (exp_fw, exp_task, exp_ft):
                    errors.append(
                        f"{json_path.name}: record has "
                        f"{r_fw}/{r_task}/{r_ft}, expected "
                        f"{exp_fw}/{exp_task}/{exp_ft}"
                    )
                    break

        return errors


# ---------------------------------------------------------------------------
# Metadata capture
# ---------------------------------------------------------------------------

def capture_full_metadata(
    model: str,
    frameworks: list[str],
    n_runs: int,
    seed: int,
    injection_probability: float,
    batch_dir: Path,
) -> dict[str, Any]:
    """Capture complete experiment metadata for reproducibility."""
    import importlib.metadata

    # Get package versions
    pkg_versions = {}
    for pkg in ["langchain", "langchain-core", "langgraph", "pyautogen",
                 "langchain-ollama", "opentelemetry-api", "scipy", "openai"]:
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pkg_versions[pkg] = "not installed"

    # Get Ollama model info
    ollama_version = "unknown"
    model_digest = "unknown"
    try:
        ov = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        ollama_version = ov.stdout.strip().replace("ollama version is ", "")
    except Exception:
        pass
    try:
        ms = subprocess.run(["ollama", "show", model], capture_output=True, text=True, timeout=5)
        for line in ms.stdout.splitlines():
            if "sha256" in line.lower():
                model_digest = line.strip()
                break
    except Exception:
        pass

    # Git info
    git_hash = "unknown"
    git_dirty = True
    try:
        gh = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                           cwd=str(batch_dir.parent.parent), timeout=5)
        git_hash = gh.stdout.strip()
        gd = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                           cwd=str(batch_dir.parent.parent), timeout=5)
        git_dirty = len(gd.stdout.strip()) > 0
    except Exception:
        pass

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "batch_dir": str(batch_dir),
        "model": model,
        "ollama_version": ollama_version,
        "model_digest": model_digest,
        "frameworks": frameworks,
        "n_runs_per_condition": n_runs,
        "seed": seed,
        "injection_probability": injection_probability,
        "failure_types": FAILURE_TYPE_NAMES,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3), 1),
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "package_versions": pkg_versions,
    }


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

@dataclass
class ConditionOutcome:
    """Result of running one experimental condition."""
    records: list[EnhancedRunRecord]
    invocation_hung: bool = False


def run_single_condition(
    framework: str,
    task: str,
    failure_type: str,
    n_runs: int,
    model: str,
    batch_id: str,
    batch_dir: Path,
    grader: ResponseGrader,
    seed: int = 42,
    failure_configs: dict[str, ToolFailureConfig | None] | None = None,
) -> ConditionOutcome:
    """Run one (framework, task, failure_type) condition.

    Returns a ConditionOutcome with records and an invocation_hung flag
    that is True if any trial timed out with a lingering thread.
    """
    configs = failure_configs or build_failure_configs(seed)
    failure_config = configs[failure_type]
    llm = build_llm(model)
    agent = build_agent(framework, llm, failure_config, model=model)

    config = BenchmarkConfig(
        framework=framework,
        task=task,
        n_runs=n_runs,
        failure_config=failure_config,
        recovery_attempts=1,
        timeout_per_run_seconds=120.0,
        seed=seed,
    )

    runner = BenchmarkRunner(agent=agent, config=config)
    results = runner.run()
    hung = runner._invocation_hung

    # Enhance results with correctness and tool evidence
    grading_config = TASK_GRADING.get(task, [])
    records: list[EnhancedRunRecord] = []

    for r in results:
        prompt_idx = r.run_id % len(grading_config) if grading_config else 0
        gc = grading_config[prompt_idx] if grading_config else {"keywords": [], "exact_answer": None}

        # Grade output
        output_text = r.output or ""
        grade = grader.grade(output_text, gc["keywords"])

        # Check exact answer if available
        exact_correct = False
        if gc["exact_answer"] and gc["exact_answer"] in output_text:
            exact_correct = True

        # Get injector stats
        injector_summary = {}
        if hasattr(agent, "get_injector_summary"):
            injector_summary = agent.get_injector_summary()

        # Get tool call count: agent.invoke() returns dict with "tool_calls" key
        # but runner stores str(output). Parse it back or use injector_calls as proxy.
        tool_calls = injector_summary.get("total_calls", 0)

        # Get the input text from the task function
        from agentprobe.benchmark.runner import _default_task_fn
        task_input = _default_task_fn(r.run_id, task)
        input_text = str(task_input.get("input", ""))

        record = EnhancedRunRecord(
            run_id=r.run_id,
            framework=framework,
            task=task,
            failure_type=failure_type,
            batch_id=batch_id,
            status=r.status,
            success=r.success,
            latency_ms=r.latency_ms,
            output=output_text,
            exception_type=r.exception_type,
            exception_msg=r.exception_msg,
            recovered=r.recovered,
            recovery_latency_ms=r.recovery_latency_ms,
            recovery_output=r.recovery_output,
            tool_call_count=tool_calls,
            injector_calls=injector_summary.get("total_calls", 0),
            injector_injections=injector_summary.get("injected_failures", 0),
            grade_score=grade.score,
            grade_correct=grade.is_correct or exact_correct,
            grade_keywords_matched=",".join(grade.matched_keywords),
            expected_answer=gc.get("exact_answer", ""),
            input_text=input_text,
            prompt_index=prompt_idx,
            failure_category=r.failure_category,
        )
        records.append(record)

    # Save per-condition results
    condition_name = f"{framework}_{task}_{failure_type}"
    csv_path = batch_dir / f"{condition_name}.csv"
    json_path = batch_dir / f"{condition_name}.json"

    if records:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].to_dict().keys())
            writer.writeheader()
            writer.writerows(r.to_dict() for r in records)

        with open(json_path, "w") as f:
            json.dump([r.to_dict() for r in records], f, indent=2)

    return ConditionOutcome(records=records, invocation_hung=hung)


def run_experiment(
    frameworks: list[str],
    tasks: list[str],
    failure_types: list[str],
    n_runs: int,
    model: str,
    batch_dir: Path,
    seed: int = 42,
) -> tuple[list[EnhancedRunRecord], bool]:
    """Run full experiment matrix with checkpointing.

    Returns (records, success). success is False if the batch was aborted
    due to a lingering invocation or a condition error that prevents
    reliable continuation.
    """
    batch_id = batch_dir.name
    state_path = batch_dir / "checkpoint.json"
    state = ExperimentState.load(state_path)
    grader = ResponseGrader(threshold=0.5)
    failure_configs = build_failure_configs(seed)

    # Validate checkpointed conditions on resume
    if state.completed_conditions:
        checkpoint_errors = state.validate_checkpoint(batch_dir, n_runs)
        if checkpoint_errors:
            _console.print("[red bold]Resume validation failed:[/red bold]")
            for err in checkpoint_errors:
                _console.print(f"  [red]• {err}[/red]")
            _console.print(
                "[red]Fix or remove the checkpoint to continue.[/red]"
            )
            return [], False

    all_records: list[EnhancedRunRecord] = []
    total_conditions = len(frameworks) * len(tasks) * len(failure_types)
    done_count = len(state.completed_conditions)

    _console.print(f"\n[bold]Experiment batch: {batch_id}[/bold]")
    _console.print(f"  Conditions: {total_conditions} ({done_count} already completed)")
    _console.print(f"  Runs per condition: {n_runs}")
    _console.print(f"  Model: {model}, Seed: {seed}")
    _console.print(f"  Injection probability: {INJECTION_PROBABILITY} (matched)")
    _console.print(f"  Output: {batch_dir}\n")

    condition_num = 0
    for framework in frameworks:
        for task in tasks:
            for ft_name in failure_types:
                condition_num += 1
                label = f"{framework}/{task}/{ft_name}"

                if state.is_done(framework, task, ft_name):
                    _console.print(f"  [{condition_num}/{total_conditions}] {label} — [dim]skipped (checkpoint)[/dim]")
                    # Load existing results
                    json_path = batch_dir / f"{framework}_{task}_{ft_name}.json"
                    if json_path.exists():
                        with open(json_path) as f:
                            existing = json.load(f)
                        all_records.extend(
                            EnhancedRunRecord(**r) for r in existing
                        )
                    continue

                _console.print(f"  [{condition_num}/{total_conditions}] {label} — running...")

                try:
                    outcome = run_single_condition(
                        framework=framework,
                        task=task,
                        failure_type=ft_name,
                        n_runs=n_runs,
                        model=model,
                        batch_id=batch_id,
                        batch_dir=batch_dir,
                        grader=grader,
                        seed=seed,
                        failure_configs=failure_configs,
                    )
                    all_records.extend(outcome.records)

                    # Abort immediately if a thread is still running
                    if outcome.invocation_hung:
                        _console.print(
                            f"    → [red]ABORT: {label} has a lingering "
                            f"invocation thread — stopping batch to prevent "
                            f"contaminated measurements[/red]"
                        )
                        return all_records, False

                    # Validate condition before checkpointing
                    executed = [r for r in outcome.records if r.status == "completed"]
                    if len(executed) < n_runs:
                        _console.print(
                            f"    → [red]ABORT: only {len(executed)}/{n_runs} "
                            f"completed — stopping batch[/red]"
                        )
                        return all_records, False

                    # Checkpoint only fully completed conditions
                    state.mark_done(framework, task, ft_name)
                    state.save(state_path)

                    # Quick summary
                    failures = sum(1 for r in executed if not r.success)
                    kw_pass = sum(1 for r in executed if r.grade_correct)
                    tool_calls = max((r.injector_calls for r in executed), default=0)
                    _console.print(
                        f"    → {len(executed)} executed, {failures} failures, "
                        f"{kw_pass}/{len(executed)} kw_pass, {tool_calls} tool calls"
                    )
                except Exception as e:
                    _console.print(
                        f"    → [red]ABORT: {type(e).__name__}: {e} — "
                        f"stopping batch[/red]"
                    )
                    return all_records, False

    return all_records, True


# ---------------------------------------------------------------------------
# Pilot validation
# ---------------------------------------------------------------------------

def run_pilot(model: str, batch_dir: Path, seed: int = 42) -> bool:
    """Run a minimal pilot to validate tool execution, correctness, and injection.

    Enforced checks (all frameworks, including AutoGen):
      1. At least one baseline run must complete successfully (status=completed, success=True).
      2. At least one baseline run must produce non-empty output.
      3. At least one injection run must complete.
      4. Injection condition must record >0 injector calls (tool exposure).

    Failure of any check prevents the full experiment from running.
    """
    _console.rule("[bold blue]Pilot Validation")

    grader = ResponseGrader(threshold=0.5)
    failure_configs = build_failure_configs(seed)
    checks_passed = True
    failures: list[str] = []

    for framework in ["langgraph", "langchain", "autogen"]:
        _console.print(f"\n[cyan]Testing {framework}...[/cyan]")

        # 1. Baseline: verify tool execution and correct answer
        baseline_outcome = run_single_condition(
            framework=framework, task="tool_use", failure_type="none",
            n_runs=2, model=model, batch_id="pilot", batch_dir=batch_dir,
            grader=grader, seed=seed, failure_configs=failure_configs,
        )

        # Abort entire pilot if a worker thread is still running
        if baseline_outcome.invocation_hung:
            msg = f"{framework}: baseline invocation hung — lingering thread detected"
            _console.print(f"  [red]FAIL: {msg}[/red]")
            failures.append(msg)
            _console.print("[red]Aborting pilot: hung invocation compromises backend state[/red]")
            return False

        executed = [r for r in baseline_outcome.records if r.status == "completed"]
        if not executed:
            msg = f"{framework}: no baseline runs completed"
            _console.print(f"  [red]FAIL: {msg}[/red]")
            failures.append(msg)
            checks_passed = False
            continue

        successes = [r for r in executed if r.success]
        if not successes:
            msg = f"{framework}: all baseline runs failed (0/{len(executed)} success)"
            _console.print(f"  [red]FAIL: {msg}[/red]")
            failures.append(msg)
            checks_passed = False
            continue

        # Parse inner answer from serialized dict (output may be str(dict))
        def _inner_output(text: str) -> str:
            if text.startswith("{"):
                try:
                    import ast
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, dict) and "output" in parsed:
                        return str(parsed["output"])
                except (ValueError, SyntaxError):
                    pass
            return text

        has_nonempty_answer = any(
            _inner_output(r.output).strip() for r in successes
        )
        if not has_nonempty_answer:
            msg = f"{framework}: baseline runs succeeded but all inner answers are empty"
            _console.print(f"  [red]FAIL: {msg}[/red]")
            failures.append(msg)
            checks_passed = False
            continue

        has_correct = any(r.grade_correct for r in successes)
        if not has_correct:
            msg = f"{framework}: baseline runs produced output but none graded correct"
            _console.print(f"  [yellow]WARN: {msg}[/yellow]")
            failures.append(msg)
            checks_passed = False

        _console.print(f"  Baseline: {len(successes)}/{len(executed)} succeeded")
        for r in successes:
            inner = _inner_output(r.output)[:80]
            _console.print(f"    Run {r.run_id}: grade={r.grade_correct}, answer={inner}...")

        # 2. Injection: verify fault exposure
        inj_outcome = run_single_condition(
            framework=framework, task="tool_use", failure_type="api_error",
            n_runs=2, model=model, batch_id="pilot", batch_dir=batch_dir,
            grader=grader, seed=seed, failure_configs=failure_configs,
        )

        # Abort entire pilot if injection left a lingering thread
        if inj_outcome.invocation_hung:
            msg = f"{framework}: injection invocation hung — lingering thread detected"
            _console.print(f"  [red]FAIL: {msg}[/red]")
            failures.append(msg)
            _console.print("[red]Aborting pilot: hung invocation compromises backend state[/red]")
            return False

        executed_inj = [r for r in inj_outcome.records if r.status == "completed"]
        if not executed_inj:
            msg = f"{framework}: no injection runs completed"
            _console.print(f"  [red]FAIL: {msg}[/red]")
            failures.append(msg)
            checks_passed = False
            continue

        # Check for tool call exposure (injector_calls is cumulative per condition)
        max_calls = max(r.injector_calls for r in executed_inj)
        max_inj = max(r.injector_injections for r in executed_inj)
        if max_calls == 0:
            msg = f"{framework}: injection condition recorded 0 tool calls — tools not exposed to injector"
            _console.print(f"  [red]FAIL: {msg}[/red]")
            failures.append(msg)
            checks_passed = False
        elif max_inj == 0:
            msg = f"{framework}: {max_calls} tool calls but 0 injected faults — injection not triggered"
            _console.print(f"  [red]FAIL: {msg}[/red]")
            failures.append(msg)
            checks_passed = False
        else:
            _console.print(f"  Injection: {max_calls} tool calls, {max_inj} injected events")

    _console.rule("[bold blue]Pilot Summary")
    if checks_passed:
        _console.print("[green]Pilot passed — safe to proceed with full matrix[/green]")
    else:
        _console.print("[red]Pilot FAILED — fix issues before full run:[/red]")
        for f in failures:
            _console.print(f"  [red]- {f}[/red]")
    return checks_passed


# ---------------------------------------------------------------------------
# Post-experiment analysis
# ---------------------------------------------------------------------------

def compute_summary(batch_dir: Path, all_records: list[EnhancedRunRecord]) -> None:
    """Compute and save aggregate metrics from raw records."""
    calc = MetricsCalculator()

    # Group by condition
    conditions: dict[str, list[dict]] = {}
    for r in all_records:
        key = f"{r.framework}/{r.task}/{r.failure_type}"
        if key not in conditions:
            conditions[key] = []
        conditions[key].append(r.to_dict())

    summary = []
    for key, runs in sorted(conditions.items()):
        fw, task, ft = key.split("/")
        metrics = calc.compute(runs, framework=fw, task=task, failure_type=ft)
        d = metrics.to_dict()

        # Add correctness metrics
        executed = [r for r in runs if r.get("status") == "completed"]
        d["n_correct"] = sum(1 for r in executed if r.get("grade_correct", False))
        d["kw_pass_rate"] = d["n_correct"] / len(executed) if executed else 0.0
        # injector_calls is cumulative per condition, not per trial — use max, not sum
        d["total_tool_calls"] = max((r.get("injector_calls", 0) for r in executed), default=0)
        d["total_injections"] = max((r.get("injector_injections", 0) for r in executed), default=0)

        summary.append(d)

    summary_path = batch_dir / "combined_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    _console.print(f"\n[green]Summary saved to {summary_path}[/green]")

    # Print results table
    table = Table(title="Experiment Results", show_lines=True)
    table.add_column("Framework", style="cyan")
    table.add_column("Task")
    table.add_column("Fault Type", style="yellow")
    table.add_column("N", justify="right")
    table.add_column("Fail Rate", justify="right")
    table.add_column("Recovery", justify="right")
    table.add_column("KW Pass", justify="right")
    table.add_column("Tool Calls", justify="right")
    table.add_column("P99 (ms)", justify="right")

    for m in summary:
        table.add_row(
            m["framework"], m["task"], m["failure_type"],
            str(m["n_runs"]),
            f"{m['failure_rate']:.1%}",
            "N/A" if m['recovery_rate'] is None else f"{m['recovery_rate']:.1%}",
            f"{m['kw_pass_rate']:.1%}",
            str(m["total_tool_calls"]),
            f"{m['p99_latency_ms']:.0f}",
        )
    _console.print(table)


# ---------------------------------------------------------------------------
# Runtime estimation
# ---------------------------------------------------------------------------

def estimate_runtime(pilot_records: list[EnhancedRunRecord], n_runs: int,
                     n_conditions: int) -> None:
    """Estimate full experiment runtime from pilot data."""
    if not pilot_records:
        return

    executed = [r for r in pilot_records if r.status == "completed"]
    if not executed:
        return

    avg_ms = sum(r.latency_ms for r in executed) / len(executed)
    # Each condition = n_runs * avg_latency + overhead
    overhead_per_condition = 5.0  # seconds for setup/teardown
    total_seconds = n_conditions * (n_runs * avg_ms / 1000 + overhead_per_condition)
    hours = total_seconds / 3600

    _console.print(f"\n[bold]Runtime estimate:[/bold]")
    _console.print(f"  Avg latency per run: {avg_ms:.0f}ms")
    _console.print(f"  Conditions: {n_conditions}")
    _console.print(f"  Runs per condition: {n_runs}")
    _console.print(f"  Estimated total: {hours:.1f} hours ({total_seconds/60:.0f} minutes)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AgentProbe Experiment Runner v2")
    parser.add_argument("--frameworks", nargs="+",
                       default=["langgraph", "langchain", "autogen"],
                       choices=["langgraph", "langchain", "autogen"])
    parser.add_argument("--tasks", nargs="+", default=["tool_use"],
                       choices=["tool_use", "rag", "multi_agent"])
    parser.add_argument("--failure-types", nargs="+",
                       default=FAILURE_TYPE_NAMES,
                       choices=FAILURE_TYPE_NAMES)
    parser.add_argument("--n-runs", type=int, default=30)
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot", action="store_true",
                       help="Run minimal pilot to validate setup")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from existing batch directory")

    args = parser.parse_args()

    # Determine batch directory
    if args.resume:
        batch_dir = Path(args.resume)
        if not batch_dir.exists():
            _console.print(f"[red]Resume directory not found: {batch_dir}[/red]")
            sys.exit(1)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_base = Path(__file__).parent / "results_v2"
        batch_dir = results_base / timestamp
        batch_dir.mkdir(parents=True, exist_ok=True)

    # Check system resources
    import shutil
    disk_free = shutil.disk_usage(str(batch_dir)).free / (1024**3)
    _console.print(f"[dim]Disk free: {disk_free:.1f} GB[/dim]")
    if disk_free < 2.0:
        _console.print("[red]Less than 2 GB disk space available. Aborting.[/red]")
        sys.exit(1)

    if args.pilot:
        pilot_dir = batch_dir / "pilot"
        pilot_dir.mkdir(exist_ok=True)
        passed = run_pilot(args.model, pilot_dir, seed=args.seed)
        if not passed:
            sys.exit(1)

        # Estimate runtime for full matrix
        pilot_records = []
        for json_file in pilot_dir.glob("*.json"):
            if json_file.name == "checkpoint.json":
                continue
            with open(json_file) as f:
                data = json.load(f)
            pilot_records.extend(EnhancedRunRecord(**r) for r in data)

        n_conditions = len(args.frameworks) * len(args.tasks) * len(args.failure_types)
        estimate_runtime(pilot_records, args.n_runs, n_conditions)
        return

    # Save metadata — preserve original on resume, reject incompatible config
    meta_path = batch_dir / "experiment_metadata.json"
    meta = capture_full_metadata(
        model=args.model,
        frameworks=args.frameworks,
        n_runs=args.n_runs,
        seed=args.seed,
        injection_probability=INJECTION_PROBABILITY,
        batch_dir=batch_dir,
    )
    if meta_path.exists():
        with open(meta_path) as f:
            original_meta = json.load(f)
        # Reject changes to critical configuration fields
        _compat_fields = [
            ("model", "model"),
            ("seed", "seed"),
            ("n_runs_per_condition", "n_runs_per_condition"),
            ("injection_probability", "injection_probability"),
        ]
        mismatches = []
        for field_name, meta_key in _compat_fields:
            orig_val = original_meta.get(meta_key)
            new_val = meta.get(meta_key)
            if orig_val is not None and orig_val != new_val:
                mismatches.append(
                    f"  {field_name}: original={orig_val}, current={new_val}"
                )
        if mismatches:
            _console.print(
                "[red bold]Resume rejected — configuration mismatch "
                "with original metadata:[/red bold]"
            )
            for m in mismatches:
                _console.print(f"[red]{m}[/red]")
            sys.exit(1)
        _console.print(
            f"[dim]Resuming — original metadata preserved at {meta_path}[/dim]"
        )
    else:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        _console.print(f"[dim]Metadata saved to {meta_path}[/dim]")

    # Run experiment
    all_records, batch_ok = run_experiment(
        frameworks=args.frameworks,
        tasks=args.tasks,
        failure_types=args.failure_types,
        n_runs=args.n_runs,
        model=args.model,
        batch_dir=batch_dir,
        seed=args.seed,
    )

    # Compute summary
    if all_records:
        compute_summary(batch_dir, all_records)

    # Save manifest
    manifest = {
        "batch_id": batch_dir.name,
        "result_files": sorted(str(p.name) for p in batch_dir.glob("*.csv")),
        "n_records": len(all_records),
        "n_conditions": len(set(
            f"{r.framework}_{r.task}_{r.failure_type}" for r in all_records
        )),
        "checksum": hashlib.sha256(
            json.dumps([r.to_dict() for r in all_records], sort_keys=True).encode()
        ).hexdigest(),
    }
    with open(batch_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if batch_ok:
        _console.print(f"\n[green]Experiment complete. Results in {batch_dir}[/green]")
    else:
        _console.print(
            f"\n[yellow]Experiment finished with errors. "
            f"Some conditions may not be checkpointed. Results in {batch_dir}[/yellow]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
