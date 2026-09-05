"""Statistical analysis of AgentProbe benchmark results.

Reads combined_summary.json (or individual framework JSONs) and produces:
  - Pairwise statistical comparisons between frameworks
  - Confidence intervals for all metrics
  - Effect sizes (Cohen's h for proportions, Cohen's d for latencies)
  - Paper-ready summary tables (LaTeX-compatible)

Usage::

    .venv/bin/python experiments/analyze_results.py
    .venv/bin/python experiments/analyze_results.py --results experiments/results/combined_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentprobe.benchmark.statistics import (
    build_comparison_report,
    FrameworkData,
    mean_ci,
    proportion_ci,
)

RESULTS_DIR = Path(__file__).parent / "results"


def load_results(path: Path | None = None) -> list[dict]:
    """Load results from JSON file."""
    if path is None:
        path = RESULTS_DIR / "combined_summary.json"
    if not path.exists():
        # Try summary.json
        path = RESULTS_DIR / "summary.json"
    if not path.exists():
        print(f"No results found at {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def analyze_failure_rates(results: list[dict]) -> None:
    """Compare failure rates across frameworks with CIs."""
    print("\n" + "=" * 70)
    print("FAILURE RATE ANALYSIS (with 95% Confidence Intervals)")
    print("=" * 70)

    # Group by framework
    frameworks = {}
    for r in results:
        fw = r["framework"]
        if fw not in frameworks:
            frameworks[fw] = {"failures": 0, "total": 0, "latencies": []}
        frameworks[fw]["failures"] += r["n_failures"]
        frameworks[fw]["total"] += r["n_runs"]
        if r["mean_latency_ms"] > 0:
            frameworks[fw]["latencies"].append(r["mean_latency_ms"])

    for fw, data in sorted(frameworks.items()):
        lo, hi = proportion_ci(data["failures"], data["total"])
        rate = data["failures"] / data["total"] if data["total"] > 0 else 0
        print(f"\n  {fw}:")
        print(f"    Overall failure rate: {rate:.1%} [{lo:.1%}, {hi:.1%}]")
        print(f"    Total runs: {data['total']}, failures: {data['failures']}")


def analyze_by_failure_type(results: list[dict]) -> None:
    """Break down failure rates by injection type."""
    print("\n" + "=" * 70)
    print("FAILURE RATES BY INJECTION TYPE")
    print("=" * 70)

    # Group by framework × failure_type
    table = {}
    for r in results:
        key = (r["framework"], r["failure_type"])
        if key not in table:
            table[key] = {"failures": 0, "total": 0, "recovery_rates": []}
        table[key]["failures"] += r["n_failures"]
        table[key]["total"] += r["n_runs"]
        if r["recovery_rate"] > 0:
            table[key]["recovery_rates"].append(r["recovery_rate"])

    # Print as table
    print(f"\n  {'Framework':<12} {'Failure Type':<18} {'Rate':>8} {'95% CI':>18} {'N':>5}")
    print("  " + "-" * 65)

    for (fw, ft), data in sorted(table.items()):
        rate = data["failures"] / data["total"] if data["total"] > 0 else 0
        lo, hi = proportion_ci(data["failures"], data["total"])
        print(f"  {fw:<12} {ft:<18} {rate:>7.1%} [{lo:.1%}, {hi:.1%}]{data['total']:>10}")


def analyze_latencies(results: list[dict]) -> None:
    """Compare latency distributions across frameworks."""
    print("\n" + "=" * 70)
    print("LATENCY ANALYSIS (ms)")
    print("=" * 70)

    # Group by framework, collect all latencies
    fw_latencies = {}
    for r in results:
        fw = r["framework"]
        if fw not in fw_latencies:
            fw_latencies[fw] = {"means": [], "p99s": []}
        if r["mean_latency_ms"] > 0:
            fw_latencies[fw]["means"].append(r["mean_latency_ms"])
            fw_latencies[fw]["p99s"].append(r["p99_latency_ms"])

    for fw, data in sorted(fw_latencies.items()):
        if not data["means"]:
            continue
        mean_lo, mean_hi = mean_ci(data["means"])
        p99_lo, p99_hi = mean_ci(data["p99s"])
        avg_mean = sum(data["means"]) / len(data["means"])
        avg_p99 = sum(data["p99s"]) / len(data["p99s"])
        print(f"\n  {fw}:")
        print(f"    Mean latency:  {avg_mean:>8.0f}ms  [{mean_lo:.0f}, {mean_hi:.0f}]")
        print(f"    Mean P99:      {avg_p99:>8.0f}ms  [{p99_lo:.0f}, {p99_hi:.0f}]")


def run_pairwise_comparisons(results: list[dict]) -> None:
    """Run full statistical comparison between frameworks."""
    print("\n" + "=" * 70)
    print("PAIRWISE STATISTICAL COMPARISONS")
    print("=" * 70)

    # Build FrameworkData for each framework
    fw_data = {}
    for r in results:
        fw = r["framework"]
        if fw not in fw_data:
            fw_data[fw] = {"successes": 0, "total": 0, "latencies": []}
        fw_data[fw]["successes"] += r["n_successes"]
        fw_data[fw]["total"] += r["n_runs"]
        if r["mean_latency_ms"] > 0:
            fw_data[fw]["latencies"].append(r["mean_latency_ms"])

    framework_data = {}
    for fw, data in fw_data.items():
        framework_data[fw] = FrameworkData(
            name=fw,
            successes=data["successes"],
            total=data["total"],
            latencies=data["latencies"],
        )

    if len(framework_data) < 2:
        print("  Need at least 2 frameworks for comparison.")
        return

    report = build_comparison_report(list(framework_data.values()))

    for comp in report.pairwise:
        print(f"\n  {comp.framework_a} vs {comp.framework_b}:")
        fr = comp.failure_rate_test
        h_val, h_mag = comp.failure_rate_effect
        print(f"    Failure rate: {fr.test_name}, p={fr.p_value:.4f}, "
              f"{'SIGNIFICANT' if fr.significant else 'not significant'}")
        print(f"    Effect size (Cohen's h): {h_val:.3f} ({h_mag})")
        if comp.latency_test:
            lt = comp.latency_test
            d_val, d_mag = comp.latency_effect
            print(f"    Latency: {lt.test_name}, p={lt.p_value:.4f}, "
                  f"{'SIGNIFICANT' if lt.significant else 'not significant'}")
            print(f"    Effect size (Cohen's d): {d_val:.3f} ({d_mag})")

    if report.omnibus_latency:
        ol = report.omnibus_latency
        print(f"\n  Omnibus latency test (all frameworks):")
        print(f"    {ol.test_name}: statistic={ol.statistic:.3f}, p={ol.p_value:.4f}, "
              f"{'SIGNIFICANT' if ol.significant else 'not significant'}")


def generate_latex_table(results: list[dict]) -> None:
    """Generate LaTeX table for the paper."""
    print("\n" + "=" * 70)
    print("LATEX TABLE (copy-paste into paper)")
    print("=" * 70)

    print(r"""
\begin{table}[h]
\caption{Benchmark results across three agent frameworks under fault injection
  (llama3.1, Ollama). Failure rate and recovery rate shown with 95\% Wilson
  score confidence intervals.}
\label{tab:results}
\begin{tabular}{llrrr}
\toprule
Framework & Failure Type & Failure Rate & Recovery Rate & P99 (ms) \\
\midrule""")

    for r in sorted(results, key=lambda x: (x["framework"], x["failure_type"])):
        fr = r["failure_rate"]
        rr = r["recovery_rate"]
        fr_lo, fr_hi = proportion_ci(r["n_failures"], r["n_runs"])
        n_recovered = int(r["recovery_rate"] * r["n_failures"]) if r["n_failures"] > 0 else 0
        rr_lo, rr_hi = proportion_ci(n_recovered, r["n_failures"]) if r["n_failures"] > 0 else (0.0, 0.0)

        fr_str = f"{fr:.0%} [{fr_lo:.0%}, {fr_hi:.0%}]"
        rr_str = f"{rr:.0%} [{rr_lo:.0%}, {rr_hi:.0%}]" if r["n_failures"] > 0 else "---"

        print(f"{r['framework']} & {r['failure_type']} & {fr_str} & {rr_str} & {r['p99_latency_ms']:.0f} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Analyze AgentProbe results")
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    results = load_results(args.results)
    print(f"Loaded {len(results)} result rows from {len(set(r['framework'] for r in results))} frameworks")

    analyze_failure_rates(results)
    analyze_by_failure_type(results)
    analyze_latencies(results)
    run_pairwise_comparisons(results)
    generate_latex_table(results)


if __name__ == "__main__":
    main()
