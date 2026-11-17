"""Sixty-second AgentProbe demo that needs no model, no key, and no network.

Run it with::

    python -m agentprobe.demo
    python -m agentprobe.demo --fault timeout --p 0.5 --runs 20
    python -m agentprobe.demo --json

It builds the LangGraph adapter around a scripted, deterministic chat model,
wraps the calculator tool with a seeded fault injector, runs a short
benchmark, and prints the reliability metrics plus the injection log, so you
can see exactly which calls were faulted and how the agent and harness reacted.
With ``--json``, the JSON report goes to stdout and progress goes to stderr.
Swap :class:`~agentprobe.fake_llm.ScriptedToolCallingModel` for a real model
(``agentprobe.llm.build_chat_model``) and nothing else changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from typing import Any

from agentprobe.benchmark.metrics import MetricsCalculator
from agentprobe.benchmark.runner import BenchmarkConfig, BenchmarkRunner
from agentprobe.fake_llm import ScriptedToolCallingModel
from agentprobe.injector.tool_failure import ToolFailureConfig, ToolFailureType
from agentprobe.observer.tracer import AgentProbeTracer

FAULTS = {
    "timeout": ToolFailureType.TIMEOUT,
    "malformed_output": ToolFailureType.MALFORMED_OUTPUT,
    "rate_limit": ToolFailureType.RATE_LIMIT,
    "api_error": ToolFailureType.API_ERROR,
}


def run_demo(
    fault: str = "rate_limit",
    probability: float = 0.3,
    n_runs: int = 10,
    seed: int = 42,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the demo and return a dict with metrics, records, and the injection summary."""
    from agentprobe.agents.langgraph_agent import LangGraphToolAgent

    # 1. Tracing. Console export off so the demo output stays readable; point
    #    otlp_endpoint at Jaeger (docker compose up) to see the spans.
    tracer = AgentProbeTracer(service_name="agentprobe-demo", console=False)

    # 2. A seeded fault injector: the same seed always faults the same calls.
    failure_config = ToolFailureConfig(
        failure_type=FAULTS[fault],
        failure_probability=probability,
        timeout_seconds=0.05,  # keep the demo fast; the paper used 5s
        retry_after_seconds=1.0,
        seed=seed,
    )

    # 3. The agent. The scripted model always calls the calculator once, then answers.
    agent = LangGraphToolAgent(llm=ScriptedToolCallingModel(), failure_config=failure_config)

    # 4. The benchmark: N runs of one calculator prompt from the paper's set
    #    (the scripted model always computes 15% of 847, so the prompt must match),
    #    one harness retry on failure. The expected answer lets us grade each run.
    expected_answer = "127.05"

    def task_fn(run_id: int, _task: str) -> dict[str, Any]:
        return {"input": "Calculate 15% of 847.", "run_id": run_id}

    config = BenchmarkConfig(
        framework="langgraph",
        task="tool_use",
        n_runs=n_runs,
        failure_config=failure_config,
        recovery_attempts=1,
        timeout_per_run_seconds=30.0,
        seed=seed,
    )
    results = BenchmarkRunner(agent=agent, config=config, task_fn=task_fn).run()
    metrics = MetricsCalculator().compute(results)
    summary = agent.get_injector_summary()
    correct = [_is_correct(r, expected_answer) for r in results if r.status == "completed"]
    n_correct = sum(correct)
    tracer.shutdown()

    if not quiet:
        _print_report(
            fault, probability, results, metrics, summary, agent, expected_answer, n_correct
        )
    return {
        "metrics": metrics,
        "results": results,
        "injector": summary,
        "n_correct": n_correct,
        "expected_answer": expected_answer,
    }


def _is_correct(result: Any, expected: str) -> bool:
    """Answer-string presence on the final answer (the output that counts).

    For a recovered run that is the recovery output; for an unrecovered
    failure there is no answer and the run is incorrect.
    """
    if result.success:
        return expected in _answer_text(result.output)
    if result.recovered:
        return expected in _answer_text(result.recovery_output)
    return False


def _print_report(
    fault: str,
    probability: float,
    results: list[Any],
    metrics: Any,
    summary: dict,
    agent: Any,
    expected_answer: str,
    n_correct: int,
) -> None:
    print()
    print(f"Fault: {fault}  p={probability}  runs={len(results)}  (seeded, reproducible)")
    print('Task: "Calculate 15% of 847."  expected answer: ' + expected_answer)
    print("-" * 64)
    print(f"tool calls made         {summary.get('total_calls', 0)}")
    print(f"faults injected         {summary.get('injected_failures', 0)}")
    print(f"failure rate            {metrics.failure_rate:.0%}  ({metrics.n_failures} runs raised)")
    rr = metrics.recovery_rate
    print(f"recovery rate           {'n/a' if rr is None else f'{rr:.0%}'}  (harness retried once)")
    print(f"MTTR                    {metrics.mttr_ms:.0f} ms")
    print(f"P50 / P99 latency       {metrics.p50_latency_ms:.0f} / {metrics.p99_latency_ms:.0f} ms")
    print(f"correct final answers   {n_correct}/{metrics.n_executed}  (answer-string presence)")
    print("-" * 64)
    print("injection log (call_index, tool, injected, outcome):")
    for rec in agent.injector.injection_log if agent.injector else []:
        outcome = type(rec.exception).__name__ if rec.exception else repr(rec.return_value)[:40]
        flag = "FAULT" if rec.injected else "ok   "
        print(f"  #{rec.call_index:<3} {rec.tool_name:<11} {flag}  {outcome}   span={rec.span_id}")
    print("-" * 64)
    print("per-run outcome (execution, final answer):")
    for r in results:
        status = "ok" if r.success else ("recovered" if r.recovered else "FAILED")
        if r.success:
            detail = _answer_text(r.output)
        elif r.recovered:
            detail = f"{r.exception_type} -> {_answer_text(r.recovery_output)}"
        else:
            detail = f"{r.exception_type} (no answer)"
        mark = "correct" if _is_correct(r, expected_answer) else "wrong  "
        print(f"  run {r.run_id:<2} {status:<9} {mark}  {detail}")
    print()
    print(
        "Next: swap ScriptedToolCallingModel for a real model via agentprobe.llm.build_chat_model,"
    )
    print("or wrap your own tools with ToolFailureInjector. See README.md.")


def _answer_text(output: str) -> str:
    """The runner stores the adapter's whole return dict as a string; show just the answer."""
    import ast

    try:
        parsed = ast.literal_eval(output)
        if isinstance(parsed, dict):
            return str(parsed.get("output", ""))[:60]
    except (ValueError, SyntaxError):
        pass
    return (output or "")[:60]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AgentProbe 60-second demo (no model needed)")
    parser.add_argument("--fault", choices=sorted(FAULTS), default="rate_limit")
    parser.add_argument("--p", type=float, default=0.3, help="injection probability per tool call")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="write the report as JSON to stdout")
    args = parser.parse_args(argv)
    if not 0.0 <= args.p <= 1.0:
        parser.error("--p must be between 0 and 1")
    if args.json:
        # The benchmark prints progress even when the demo's text report is disabled.
        with redirect_stdout(sys.stderr):
            report = run_demo(args.fault, args.p, args.runs, args.seed, quiet=True)
        payload = {
            "metrics": report["metrics"].to_dict(),
            "injector": report["injector"],
            "results": [result.to_dict() for result in report["results"]],
        }
        print(json.dumps(payload, allow_nan=False))
    else:
        run_demo(args.fault, args.p, args.runs, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
