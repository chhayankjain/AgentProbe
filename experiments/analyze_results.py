"""Statistical analysis of AgentProbe benchmark results.

Loads per-run CSV files from experiments/results/ for raw latency data,
and combined_summary.json for aggregate counts.  Produces:
  - Pairwise statistical comparisons between frameworks
  - Confidence intervals for all metrics
  - Effect sizes (Cohen's h for proportions, Cohen's d for latencies)
  - Paper-ready summary tables (LaTeX-compatible)

Usage::

    .venv/bin/python experiments/analyze_results.py
    .venv/bin/python experiments/analyze_results.py --results experiments/results/combined_summary.json
"""

from __future__ import annotations

import csv
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


def load_raw_runs() -> list[dict]:
    """Load all per-run records from individual CSV files.

    Returns a flat list of dicts with keys: framework, task, failure_type,
    success, latency_ms, recovered, recovery_latency_ms, etc.
    """
    rows: list[dict] = []
    for csv_path in sorted(RESULTS_DIR.glob("*.csv")):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Coerce types
                row["latency_ms"] = float(row.get("latency_ms", 0))
                row["recovery_latency_ms"] = float(row.get("recovery_latency_ms", 0))
                row["success"] = row.get("success", "").strip() == "True"
                row["recovered"] = row.get("recovered", "").strip() == "True"
                rows.append(row)
    return rows


def analyze_failure_rates(results: list[dict]) -> None:
    """Compare failure rates across frameworks with CIs (injection only, excluding baseline)."""
    print("\n" + "=" * 70)
    print("FAILURE RATE ANALYSIS — INJECTION ONLY (with 95% Confidence Intervals)")
    print("=" * 70)

    # Separate baseline and injection
    injection_results = [r for r in results if r["failure_type"] != "none"]
    baseline_results = [r for r in results if r["failure_type"] == "none"]

    for label, subset in [("Injection", injection_results), ("Baseline", baseline_results)]:
        print(f"\n  --- {label} ---")
        frameworks = {}
        for r in subset:
            fw = r["framework"]
            if fw not in frameworks:
                frameworks[fw] = {"failures": 0, "total": 0}
            frameworks[fw]["failures"] += r["n_failures"]
            frameworks[fw]["total"] += r["n_runs"]

        for fw, data in sorted(frameworks.items()):
            lo, hi = proportion_ci(data["failures"], data["total"])
            rate = data["failures"] / data["total"] if data["total"] > 0 else 0
            print(f"\n    {fw}:")
            print(f"      Failure rate: {rate:.1%} [{lo:.1%}, {hi:.1%}]")
            print(f"      Runs: {data['total']}, failures: {data['failures']}")


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


def analyze_latencies(raw_runs: list[dict]) -> None:
    """Compare latency distributions across frameworks using per-run raw data."""
    print("\n" + "=" * 70)
    print("LATENCY ANALYSIS (ms) — baseline only, per-run data")
    print("=" * 70)

    # Use baseline conditions only (uncontaminated by injection)
    baseline = [r for r in raw_runs if r["failure_type"] == "none"]

    fw_latencies: dict[str, list[float]] = {}
    for r in baseline:
        fw = r["framework"]
        if fw not in fw_latencies:
            fw_latencies[fw] = []
        fw_latencies[fw].append(r["latency_ms"])

    for fw, latencies in sorted(fw_latencies.items()):
        if not latencies:
            continue
        lo, hi = mean_ci(latencies)
        avg = sum(latencies) / len(latencies)
        latencies_sorted = sorted(latencies)
        p99_idx = min(int(len(latencies_sorted) * 0.99), len(latencies_sorted) - 1)
        p99 = latencies_sorted[p99_idx]
        print(f"\n  {fw} (n={len(latencies)} runs):")
        print(f"    Mean latency:  {avg:>8.0f}ms  [{lo:.0f}, {hi:.0f}]")
        print(f"    P99 latency:   {p99:>8.0f}ms")


def analyze_baseline_latency_comparison(raw_runs: list[dict]) -> None:
    """Omnibus latency test on baseline runs — matches the paper's reported H statistic."""
    print("\n" + "=" * 70)
    print("BASELINE LATENCY OMNIBUS TEST (Kruskal-Wallis)")
    print("=" * 70)

    baseline = [r for r in raw_runs if r["failure_type"] == "none"]
    fw_latencies: dict[str, list[float]] = {}
    for r in baseline:
        fw = r["framework"]
        if fw not in fw_latencies:
            fw_latencies[fw] = []
        fw_latencies[fw].append(r["latency_ms"])

    if len(fw_latencies) < 2:
        print("  Need at least 2 frameworks for comparison.")
        return

    from scipy import stats
    groups = list(fw_latencies.values())
    names = list(fw_latencies.keys())
    h_stat, p_val = stats.kruskal(*groups)
    sig = "SIGNIFICANT" if p_val < 0.05 else "not significant"
    total_n = sum(len(g) for g in groups)
    print(f"\n  Frameworks: {', '.join(f'{n} (n={len(fw_latencies[n])})' for n in names)}")
    print(f"  Total baseline runs: {total_n}")
    print(f"  Kruskal-Wallis H = {h_stat:.6f}, p = {p_val:.6f}, {sig}")


def run_pairwise_comparisons(results: list[dict], raw_runs: list[dict]) -> None:
    """Run full statistical comparison between frameworks (injection conditions only)."""
    print("\n" + "=" * 70)
    print("PAIRWISE STATISTICAL COMPARISONS — INJECTION ERROR RATES")
    print("=" * 70)

    # Use summary for counts, raw runs for latencies
    injection_summary = [r for r in results if r["failure_type"] != "none"]
    injection_raw = [r for r in raw_runs if r["failure_type"] != "none"]

    # Build FrameworkData — counts from summary, latencies from raw
    fw_data: dict[str, dict] = {}
    for r in injection_summary:
        fw = r["framework"]
        if fw not in fw_data:
            fw_data[fw] = {"successes": 0, "total": 0, "latencies": []}
        fw_data[fw]["successes"] += r["n_successes"]
        fw_data[fw]["total"] += r["n_runs"]

    for r in injection_raw:
        fw = r["framework"]
        if fw in fw_data:
            fw_data[fw]["latencies"].append(r["latency_ms"])

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
    raw_runs = load_raw_runs()
    print(f"Loaded {len(results)} summary rows from {len(set(r['framework'] for r in results))} frameworks")
    print(f"Loaded {len(raw_runs)} per-run records from CSV files")

    analyze_failure_rates(results)
    analyze_by_failure_type(results)
    analyze_latencies(raw_runs)
    analyze_baseline_latency_comparison(raw_runs)
    run_pairwise_comparisons(results, raw_runs)
    generate_latex_table(results)


if __name__ == "__main__":
    main()
