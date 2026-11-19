"""Combine per-framework benchmark results into a single summary.

Reads all summary.json files from experiments/results/ and produces
combined_summary.json with all frameworks merged.

Usage::

    .venv/bin/python experiments/combine_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    summary_path = RESULTS_DIR / "summary.json"
    if not summary_path.exists():
        print(f"No summary.json found at {summary_path}")
        sys.exit(1)

    with open(summary_path) as f:
        all_metrics = json.load(f)

    print(f"Loaded {len(all_metrics)} result rows")

    # Count by framework
    frameworks = {}
    for m in all_metrics:
        fw = m["framework"]
        if fw not in frameworks:
            frameworks[fw] = 0
        frameworks[fw] += m["n_runs"]

    for fw, n in sorted(frameworks.items()):
        n_conditions = sum(1 for m in all_metrics if m["framework"] == fw)
        print(f"  {fw}: {n} total runs across {n_conditions} conditions")

    # Save as combined_summary.json
    combined_path = RESULTS_DIR / "combined_summary.json"
    with open(combined_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved to {combined_path}")


if __name__ == "__main__":
    main()
