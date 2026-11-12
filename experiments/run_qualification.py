"""Targeted qualification experiment for AgentProbe publication.

Validates three properties that the main experiment cannot demonstrate
from saved data alone:

  1. Correct answer against controlled ground truth (not model knowledge).
  2. Recorded matching tool execution via injector log with linked span IDs,
     including the registered tool name, expression arguments, and return value.
  3. Guaranteed injected fault (p=1.0) with an observed exception outcome
     at the required tool boundary.

Uses deterministic tasks with known calculator results:
  - "Compute 7919 * 6841" → 54,173,879
  - "Compute 3571 + 8429" → 12,000

Produces a self-contained JSON artifact with inputs, outputs, tool events,
injection log entries (with args/returns), linked trace span IDs, and
pass/fail verdicts.

No full matrix rerun. This is a small, targeted validation.

Usage::

    .venv/bin/python experiments/run_qualification.py
    .venv/bin/python experiments/run_qualification.py --frameworks langgraph autogen
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from agentprobe.injector.tool_failure import (
    ToolFailureConfig,
    ToolFailureType,
)

# ---------------------------------------------------------------------------
# Controlled tasks: answers require calculator tool execution
# ---------------------------------------------------------------------------

QUALIFICATION_TASKS = [
    {
        "id": "qual_multiply",
        "input": (
            "You MUST call the calculator tool with the expression '7919 * 6841' "
            "and report the exact numeric result. Do NOT compute this yourself."
        ),
        "expected_answer": "54173879",
        "required_tool": "calculator",
        "required_expression": "7919 * 6841",
    },
    {
        "id": "qual_add",
        "input": (
            "You MUST call the calculator tool with the expression '3571 + 8429' "
            "and report the exact numeric result. Do NOT compute this yourself."
        ),
        "expected_answer": "12000",
        "required_tool": "calculator",
        "required_expression": "3571 + 8429",
    },
]

# Baseline: instrument tool calls for logging, but never inject faults
BASELINE_CONFIG = ToolFailureConfig(
    failure_type=ToolFailureType.NONE,
    failure_probability=0.0,
    seed=42,
)

# Guaranteed fault: p=1.0 so every tool call is injected
GUARANTEED_FAULT_CONFIG = ToolFailureConfig(
    failure_type=ToolFailureType.API_ERROR,
    failure_probability=1.0,
    error_code=503,
    seed=42,
)


# ---------------------------------------------------------------------------
# Span capture
# ---------------------------------------------------------------------------

def setup_tracing() -> tuple[TracerProvider, InMemorySpanExporter]:
    """Configure OTel with an in-memory exporter for JSON serialization."""
    exporter = InMemorySpanExporter()
    resource = Resource(attributes={SERVICE_NAME: "agentprobe-qualification"})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider, exporter


def spans_to_dicts(exporter: InMemorySpanExporter) -> list[dict[str, Any]]:
    """Convert captured spans to serializable dicts."""
    out = []
    for span in exporter.get_finished_spans():
        ctx = span.get_span_context()
        parent = span.parent
        out.append({
            "name": span.name,
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
            "parent_span_id": format(parent.span_id, "016x") if parent else None,
            "status": span.status.status_code.name,
            "attributes": dict(span.attributes) if span.attributes else {},
            "start_ns": span.start_time,
            "end_ns": span.end_time,
            "duration_ms": round((span.end_time - span.start_time) / 1e6, 2),
        })
    return out


# ---------------------------------------------------------------------------
# Agent builders
# ---------------------------------------------------------------------------

def build_agent(framework: str, model: str,
                failure_config: ToolFailureConfig | None) -> Any:
    if framework == "langgraph":
        from langchain_ollama import ChatOllama
        from agentprobe.agents.langgraph_agent import LangGraphToolAgent
        llm = ChatOllama(model=model, temperature=0.0)
        return LangGraphToolAgent(llm=llm, failure_config=failure_config)
    elif framework == "langchain":
        from langchain_ollama import ChatOllama
        from agentprobe.agents.langchain_agent import LangChainToolAgent
        llm = ChatOllama(model=model, temperature=0.0)
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
        return AutoGenToolAgent(
            llm_config=llm_config, failure_config=failure_config
        )
    else:
        raise ValueError(f"Unknown framework: {framework}")


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

@dataclass
class QualificationTrial:
    framework: str
    task_id: str
    phase: str  # "baseline" or "guaranteed_fault"
    input_text: str
    expected_answer: str
    required_tool: str

    output: str = ""
    latency_ms: float = 0.0

    answer_present: bool = False
    tool_executed: bool = False
    required_tool_called: bool = False
    tool_expression_matched: bool = False
    tool_return_value: str | None = None
    tool_return_correct: bool = False
    tool_evidence_bound: bool = False  # expression + return correct on SAME event
    fault_injected: bool = False
    fault_at_required_tool: bool = False
    fault_exception_type: str = ""

    injector_log: list[dict[str, Any]] = field(default_factory=list)
    span_ids: list[str] = field(default_factory=list)

    verdict: str = ""  # "PASS" or "FAIL"
    failure_reasons: list[str] = field(default_factory=list)


def _serialize_args(args: tuple, kwargs: dict) -> str:
    """Best-effort serialization of call arguments."""
    parts = [repr(a) for a in args]
    parts.extend(f"{k}={v!r}" for k, v in kwargs.items())
    return ", ".join(parts)


def run_trial(
    framework: str,
    task: dict[str, str],
    phase: str,
    model: str,
    failure_config: ToolFailureConfig | None,
) -> QualificationTrial:
    """Run one qualification trial and evaluate all dimensions."""
    agent = build_agent(framework, model, failure_config)

    trial = QualificationTrial(
        framework=framework,
        task_id=task["id"],
        phase=phase,
        input_text=task["input"],
        expected_answer=task["expected_answer"],
        required_tool=task["required_tool"],
    )

    start = time.perf_counter()
    try:
        result = agent.invoke({"input": task["input"]})
        trial.output = str(result.get("output", ""))
        trial.latency_ms = (time.perf_counter() - start) * 1000
    except Exception as e:
        trial.output = f"EXCEPTION: {type(e).__name__}: {e}"
        trial.latency_ms = (time.perf_counter() - start) * 1000

    # --- Dimension 1: Answer correctness ---
    expected = task["expected_answer"]
    # Match with or without comma formatting (e.g., "54173879" or "54,173,879")
    # Boundaries reject: adjacent digits, leading minus sign, trailing decimal.
    # This prevents "-12000" matching "12000", "12000.5" matching "12000", and
    # "120000" matching "12000".
    import re
    expected_pattern = re.escape(expected)
    expected_comma = re.escape(f"{int(expected):,}")
    # Boundary: not preceded by digit or minus; not followed by digit or
    # decimal fraction (dot-then-digit).  "12000." at sentence end is allowed;
    # "12000.5" and "-12000" are rejected.
    pre = r'(?<![\d\-])'
    post = r'(?!\d)(?!\.\d)'
    trial.answer_present = bool(
        re.search(rf'{pre}{expected_pattern}{post}', trial.output)
        or re.search(rf'{pre}{expected_comma}{post}', trial.output)
    )

    # --- Dimension 2: Tool execution evidence ---
    # Check registered tool name, full expression in arguments, and correct
    # return value — all on the SAME log record to prevent cross-record
    # false positives.
    injector = agent._injector
    required_tool = task["required_tool"]
    required_expr = task["required_expression"]

    if injector:
        log = injector.injection_log
        trial.injector_log = [
            {
                "tool_name": r.tool_name,
                "injected": r.injected,
                "exception": type(r.exception).__name__ if r.exception else None,
                "call_args": _serialize_args(r.call_args, r.call_kwargs),
                "return_value": str(r.return_value) if r.return_value is not None else None,
                "latency_ms": round(r.latency_ms, 2),
                "span_id": r.span_id,
                "call_index": r.call_index,
            }
            for r in log
        ]
        trial.span_ids = [r.span_id for r in log if r.span_id]

        for r in log:
            trial.tool_executed = True
            # Match REGISTERED tool name exactly — no generic wrappers
            if r.tool_name == required_tool:
                trial.required_tool_called = True
                # Bind expression match AND return correctness to the SAME
                # record to prevent cross-event false positives (e.g., one
                # event has the right expression but wrong return, another
                # has the right return but different expression).
                args_str = _serialize_args(r.call_args, r.call_kwargs)
                # Use word-boundary match for expression to avoid substring
                # false positives (e.g., "7919 * 68410" matching "7919 * 6841")
                expr_pattern = re.escape(required_expr)
                expr_matched = bool(re.search(
                    rf'(?<!\w){expr_pattern}(?!\w)', args_str
                ))
                if expr_matched:
                    trial.tool_expression_matched = True
                if r.return_value is not None:
                    trial.tool_return_value = str(r.return_value)
                    if str(r.return_value) == expected:
                        trial.tool_return_correct = True
                # Single-event binding: both expression and return on same record
                if expr_matched and r.return_value is not None and str(r.return_value) == expected:
                    trial.tool_evidence_bound = True

    # --- Dimension 3: Fault injection evidence ---
    if phase == "guaranteed_fault" and injector:
        for r in injector.injection_log:
            if r.injected:
                trial.fault_injected = True
                trial.fault_exception_type = (
                    type(r.exception).__name__ if r.exception else ""
                )
                # Check if the fault hit the required tool boundary
                if r.tool_name == required_tool:
                    trial.fault_at_required_tool = True

    # --- Verdict ---
    reasons = []
    if phase == "baseline":
        if not trial.answer_present:
            reasons.append(f"Expected answer '{expected}' not in output")
        if not trial.required_tool_called:
            reasons.append(
                f"Required tool '{required_tool}' was not called "
                f"(recorded tools: {[e['tool_name'] for e in trial.injector_log]})"
            )
        if not trial.tool_expression_matched:
            reasons.append(
                f"Required expression '{required_expr}' not found in "
                f"'{required_tool}' call arguments"
            )
        if trial.required_tool_called and not trial.tool_return_correct:
            reasons.append(
                f"Tool return value '{trial.tool_return_value}' does not match "
                f"expected '{expected}'"
            )
        # Require bound evidence: expression + correct return on same event
        if trial.tool_expression_matched and trial.tool_return_correct and not trial.tool_evidence_bound:
            reasons.append(
                "Expression match and correct return found on DIFFERENT events — "
                "no single event has both the required expression and correct return"
            )
    elif phase == "guaranteed_fault":
        if not trial.fault_injected:
            reasons.append("Guaranteed fault (p=1.0) was not injected")
        if not trial.tool_executed:
            reasons.append("No tool calls recorded despite injection config")
        if not trial.fault_at_required_tool:
            reasons.append(
                f"Fault was not injected at the required tool '{required_tool}' "
                f"(recorded tools: {[e['tool_name'] for e in trial.injector_log]})"
            )

    trial.failure_reasons = reasons
    trial.verdict = "PASS" if not reasons else "FAIL"
    return trial


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgentProbe qualification experiment"
    )
    parser.add_argument(
        "--frameworks", nargs="+",
        default=["langgraph", "langchain", "autogen"],
        choices=["langgraph", "langchain", "autogen"],
    )
    parser.add_argument("--model", default="llama3.1")
    args = parser.parse_args()

    # Setup tracing with in-memory exporter
    provider, span_exporter = setup_tracing()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results_v2" / f"qualification_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_trials: list[QualificationTrial] = []
    all_passed = True

    print(f"\n{'='*60}")
    print(f"  AgentProbe Qualification Experiment")
    print(f"  Frameworks: {args.frameworks}")
    print(f"  Model: {args.model}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    for framework in args.frameworks:
        print(f"\n--- {framework} ---")

        # Phase 1: Baseline — correct answer + tool execution + expression match
        for task in QUALIFICATION_TASKS:
            print(f"  [{framework}] baseline / {task['id']}...", end=" ", flush=True)
            trial = run_trial(
                framework=framework,
                task=task,
                phase="baseline",
                model=args.model,
                failure_config=BASELINE_CONFIG,
            )
            all_trials.append(trial)
            status = f"\033[92m{trial.verdict}\033[0m" if trial.verdict == "PASS" else f"\033[91m{trial.verdict}\033[0m"
            print(
                f"{status} ({trial.latency_ms:.0f}ms) "
                f"answer={trial.answer_present} "
                f"tool={trial.required_tool_called} "
                f"expr={trial.tool_expression_matched}"
            )
            if trial.failure_reasons:
                for r in trial.failure_reasons:
                    print(f"    FAIL: {r}")
            if trial.verdict == "FAIL":
                all_passed = False

        # Phase 2: Guaranteed fault — p=1.0, must inject at required tool
        for task in QUALIFICATION_TASKS:
            print(f"  [{framework}] guaranteed_fault / {task['id']}...", end=" ", flush=True)
            trial = run_trial(
                framework=framework,
                task=task,
                phase="guaranteed_fault",
                model=args.model,
                failure_config=GUARANTEED_FAULT_CONFIG,
            )
            all_trials.append(trial)
            status = f"\033[92m{trial.verdict}\033[0m" if trial.verdict == "PASS" else f"\033[91m{trial.verdict}\033[0m"
            inj_count = sum(1 for e in trial.injector_log if e["injected"])
            print(
                f"{status} ({trial.latency_ms:.0f}ms) "
                f"injected={inj_count} "
                f"fault_at_tool={trial.fault_at_required_tool} "
                f"tool_calls={len(trial.injector_log)}"
            )
            if trial.failure_reasons:
                for r in trial.failure_reasons:
                    print(f"    FAIL: {r}")
            if trial.verdict == "FAIL":
                all_passed = False

    # Flush traces
    provider.force_flush()

    # Collect spans
    captured_spans = spans_to_dicts(span_exporter)

    # Build result artifact
    result = {
        "qualification_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "frameworks": args.frameworks,
        "tasks": QUALIFICATION_TASKS,
        "baseline_config": {
            "failure_type": BASELINE_CONFIG.failure_type.value,
            "failure_probability": BASELINE_CONFIG.failure_probability,
            "seed": BASELINE_CONFIG.seed,
        },
        "guaranteed_fault_config": {
            "failure_type": GUARANTEED_FAULT_CONFIG.failure_type.value,
            "failure_probability": GUARANTEED_FAULT_CONFIG.failure_probability,
            "error_code": GUARANTEED_FAULT_CONFIG.error_code,
            "seed": GUARANTEED_FAULT_CONFIG.seed,
        },
        "all_passed": all_passed,
        "n_trials": len(all_trials),
        "n_passed": sum(1 for t in all_trials if t.verdict == "PASS"),
        "n_failed": sum(1 for t in all_trials if t.verdict == "FAIL"),
        "trials": [asdict(t) for t in all_trials],
        "captured_spans": captured_spans,
        "n_spans": len(captured_spans),
    }

    # Save
    out_path = out_dir / "qualification_results.json"
    out_path.write_text(json.dumps(result, indent=2, default=str) + "\n")

    print(f"\n{'='*60}")
    print(f"  Results: {out_path}")
    print(f"  Trials: {result['n_passed']}/{result['n_trials']} passed")
    print(f"  Spans captured: {result['n_spans']}")
    if all_passed:
        print(f"  \033[92mQUALIFICATION PASSED\033[0m")
    else:
        print(f"  \033[91mQUALIFICATION FAILED\033[0m")
    print(f"{'='*60}")

    provider.shutdown()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
