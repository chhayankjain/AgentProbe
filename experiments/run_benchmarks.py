"""Main benchmark runner script for AgentProbe experiments.

Runs the full benchmark matrix:
  - Frameworks: LangGraph, LangChain, AutoGen
  - Tasks: tool_use, rag, multi_agent
  - Failure types: none (baseline), timeout, malformed_output, rate_limit, api_error

Results are saved to experiments/results/ as CSV and JSON.

Usage::

    # Baseline only (no failures)
    python experiments/run_benchmarks.py --frameworks langgraph --tasks tool_use --baseline-only

    # Full matrix with Ollama/Llama-3
    python experiments/run_benchmarks.py --llm ollama --model llama3

    # Quick smoke test (5 runs per config)
    python experiments/run_benchmarks.py --n-runs 5 --frameworks langgraph --tasks tool_use

    # Save results for paper
    python experiments/run_benchmarks.py --n-runs 50 --save-results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure agentprobe package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from agentprobe.benchmark.metrics import MetricsCalculator
from agentprobe.benchmark.runner import BenchmarkConfig, BenchmarkRunner
from agentprobe.injector.tool_failure import ToolFailureConfig, ToolFailureType
from agentprobe.observer.tracer import AgentProbeTracer

_console = Console()

RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Failure configurations
# ---------------------------------------------------------------------------

FAILURE_CONFIGS: dict[str, ToolFailureConfig | None] = {
    "none": None,
    "timeout": ToolFailureConfig(
        failure_type=ToolFailureType.TIMEOUT,
        failure_probability=0.2,
        timeout_seconds=5.0,
        seed=42,
    ),
    "malformed_output": ToolFailureConfig(
        failure_type=ToolFailureType.MALFORMED_OUTPUT,
        failure_probability=0.2,
        seed=42,
    ),
    "rate_limit": ToolFailureConfig(
        failure_type=ToolFailureType.RATE_LIMIT,
        failure_probability=0.15,
        retry_after_seconds=1.0,  # Short for benchmarking
        seed=42,
    ),
    "api_error": ToolFailureConfig(
        failure_type=ToolFailureType.API_ERROR,
        failure_probability=0.15,
        error_code=503,
        seed=42,
    ),
}


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def build_llm(llm_backend: str, model: str) -> Any:
    """Build a LangChain LLM from CLI arguments."""
    if llm_backend == "ollama":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model, temperature=0.0)
        except ImportError as e:
            _console.print(
                "[red]langchain-ollama not installed. "
                "Run: pip install 'agentprobe[ollama]'[/red]"
            )
            raise SystemExit(1) from e

    elif llm_backend == "openai":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=0.0)
        except ImportError as e:
            _console.print("[red]langchain-openai not installed.[/red]")
            raise SystemExit(1) from e

    elif llm_backend == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, temperature=0.0)
        except ImportError as e:
            _console.print("[red]langchain-anthropic not installed.[/red]")
            raise SystemExit(1) from e

    else:
        raise ValueError(f"Unknown LLM backend: {llm_backend}")


def build_agent(framework: str, llm: Any, failure_config: ToolFailureConfig | None) -> Any:
    if framework == "langgraph":
        from agentprobe.agents.langgraph_agent import LangGraphToolAgent
        return LangGraphToolAgent(llm=llm, failure_config=failure_config)
    elif framework == "langchain":
        from agentprobe.agents.langchain_agent import LangChainToolAgent
        return LangChainToolAgent(llm=llm, failure_config=failure_config)
    elif framework == "autogen":
        from agentprobe.agents.autogen_agent import AutoGenToolAgent
        llm_config = {
            "config_list": [{"model": "llama3.1", "base_url": "http://localhost:11434/v1", "api_key": "ollama"}]
        }
        return AutoGenToolAgent(llm_config=llm_config, failure_config=failure_config)
    else:
        raise ValueError(f"Unknown framework: {framework}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_matrix(
    frameworks: list[str],
    tasks: list[str],
    failure_types: list[str],
    n_runs: int,
    llm_backend: str,
    model: str,
    save: bool,
) -> list[dict[str, Any]]:
    """Run the full benchmark matrix and return aggregated metrics."""
    tracer = AgentProbeTracer(service_name="agentprobe-benchmark", console=False)
    calc = MetricsCalculator()
    all_metrics = []

    for framework in frameworks:
        llm = build_llm(llm_backend, model)
        for task in tasks:
            for ft_name in failure_types:
                failure_config = FAILURE_CONFIGS[ft_name]
                agent = build_agent(framework, llm, failure_config)

                save_path = (
                    str(RESULTS_DIR / f"{framework}_{task}_{ft_name}")
                    if save
                    else None
                )

                config = BenchmarkConfig(
                    framework=framework,
                    task=task,
                    n_runs=n_runs,
                    failure_config=failure_config,
                    recovery_attempts=1,
                    save_results_path=save_path,
                    seed=42,
                )

                runner = BenchmarkRunner(agent=agent, config=config)
                runs = runner.run()
                metrics = calc.compute(runs, framework=framework, task=task, failure_type=ft_name)
                all_metrics.append(metrics.to_dict())

    tracer.shutdown()
    return all_metrics


def print_results_table(all_metrics: list[dict[str, Any]]) -> None:
    table = Table(title="AgentProbe Benchmark Results", show_lines=True)
    table.add_column("Framework", style="cyan")
    table.add_column("Task", style="green")
    table.add_column("Failure Type", style="yellow")
    table.add_column("Failure Rate", justify="right")
    table.add_column("Recovery Rate", justify="right")
    table.add_column("MTTR (ms)", justify="right")
    table.add_column("P99 (ms)", justify="right")

    for m in all_metrics:
        table.add_row(
            m["framework"],
            m["task"],
            m["failure_type"],
            f"{m['failure_rate']:.1%}",
            f"{m['recovery_rate']:.1%}",
            f"{m['mttr_ms']:.0f}",
            f"{m['p99_latency_ms']:.0f}",
        )
    _console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentProbe Benchmark Runner")
    parser.add_argument(
        "--frameworks", nargs="+",
        default=["langgraph"],
        choices=["langgraph", "langchain", "autogen"],
    )
    parser.add_argument(
        "--tasks", nargs="+",
        default=["tool_use"],
        choices=["tool_use", "rag", "multi_agent"],
    )
    parser.add_argument(
        "--failure-types", nargs="+",
        default=["none", "timeout", "malformed_output"],
        choices=list(FAILURE_CONFIGS.keys()),
    )
    parser.add_argument("--n-runs", type=int, default=30)
    parser.add_argument("--llm", default="ollama", choices=["ollama", "openai", "anthropic"])
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--save-results", action="store_true")

    args = parser.parse_args()

    if args.baseline_only:
        failure_types = ["none"]
    else:
        failure_types = args.failure_types

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = run_matrix(
        frameworks=args.frameworks,
        tasks=args.tasks,
        failure_types=failure_types,
        n_runs=args.n_runs,
        llm_backend=args.llm,
        model=args.model,
        save=args.save_results,
    )

    print_results_table(all_metrics)

    if args.save_results:
        summary_path = RESULTS_DIR / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        _console.print(f"\n[green]Summary saved to {summary_path}[/green]")


if __name__ == "__main__":
    main()
