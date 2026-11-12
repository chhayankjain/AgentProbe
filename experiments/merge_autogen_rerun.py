"""Merge new AutoGen rerun results into combined_summary.json.

Reads raw per-condition JSON files from experiments/results/autogen_*.json,
recomputes summary metrics, and replaces the AutoGen entries in
combined_summary.json while preserving LangGraph/LangChain data.

Usage::

    .venv/bin/python experiments/merge_autogen_rerun.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
COMBINED_PATH = RESULTS_DIR / "combined_summary.json"


def compute_metrics(runs: list[dict], framework: str, task: str, failure_type: str) -> dict:
    """Compute summary metrics from raw run data."""
    n_runs = len(runs)
    failures = [r for r in runs if not r["success"]]
    successes = [r for r in runs if r["success"]]
    n_failures = len(failures)
    n_successes = len(successes)
    failure_rate = n_failures / n_runs if n_runs else 0.0

    # Recovery
    recovered = [r for r in failures if r.get("recovered", False)]
    recovery_rate = len(recovered) / n_failures if n_failures else 0.0
    mttr_ms = float(np.mean([r["recovery_latency_ms"] for r in recovered])) if recovered else 0.0

    # Latency (all runs, matching original calculator behavior)
    all_latencies = [r["latency_ms"] for r in runs]
    if all_latencies:
        mean_lat = float(np.mean(all_latencies))
        p50 = float(np.percentile(all_latencies, 50))
        p90 = float(np.percentile(all_latencies, 90))
        p99 = float(np.percentile(all_latencies, 99))
        max_lat = float(np.max(all_latencies))
    else:
        mean_lat = p50 = p90 = p99 = max_lat = 0.0

    return {
        "framework": framework,
        "task": task,
        "failure_type": failure_type,
        "n_runs": n_runs,
        "failure_rate": round(failure_rate, 4),
        "n_failures": n_failures,
        "n_successes": n_successes,
        "recovery_rate": round(recovery_rate, 4),
        "mttr_ms": round(mttr_ms, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "p50_latency_ms": round(p50, 2),
        "p90_latency_ms": round(p90, 2),
        "p99_latency_ms": round(p99, 2),
        "max_latency_ms": round(max_lat, 2),
    }


def main() -> None:
    # Load existing combined summary
    with open(COMBINED_PATH) as f:
        combined = json.load(f)

    # Keep only non-autogen entries
    non_autogen = [r for r in combined if r["framework"] != "autogen"]
    print(f"Kept {len(non_autogen)} non-AutoGen entries")

    # Process all autogen result files
    tasks = ["tool_use", "rag", "multi_agent"]
    failure_types = ["none", "timeout", "malformed_output", "rate_limit", "api_error"]
    new_autogen = []
    stale_count = 0

    for task in tasks:
        for ft in failure_types:
            fname = f"autogen_{task}_{ft}.json"
            fpath = RESULTS_DIR / fname
            if not fpath.exists():
                print(f"  MISSING: {fname}")
                continue

            with open(fpath) as f:
                runs = json.load(f)

            # Check if this is fresh data (real latencies > 1s) or stale cached data (<100ms)
            mean_lat = np.mean([r["latency_ms"] for r in runs])
            is_stale = mean_lat < 100  # cached responses are <100ms

            metrics = compute_metrics(runs, "autogen", task, ft)
            new_autogen.append(metrics)

            status = "STALE (cached)" if is_stale else "OK"
            if is_stale:
                stale_count += 1
            print(f"  {fname}: {metrics['n_failures']}/{metrics['n_runs']} failures, "
                  f"mean={metrics['mean_latency_ms']:.0f}ms [{status}]")

    # Merge
    merged = non_autogen + new_autogen
    merged.sort(key=lambda r: (r["framework"], r["task"], r["failure_type"]))

    with open(COMBINED_PATH, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerged {len(new_autogen)} AutoGen entries into combined_summary.json")
    print(f"Total entries: {len(merged)}")
    if stale_count:
        print(f"WARNING: {stale_count} conditions still have stale cached data. "
              f"Re-run this script after the benchmark finishes.")

    # Print overall AutoGen summary
    total_runs = sum(m["n_runs"] for m in new_autogen)
    total_failures = sum(m["n_failures"] for m in new_autogen)
    print(f"\nAutoGen overall: {total_failures}/{total_runs} failures "
          f"({total_failures/total_runs*100:.1f}%)")


if __name__ == "__main__":
    main()
