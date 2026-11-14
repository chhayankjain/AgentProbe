"""Analyze new Qwen AutoGen rerun data and print updated Table 7 values.

Run after the Qwen AutoGen rerun completes:

    .venv/bin/python experiments/update_qwen_autogen.py experiments/results_v2/20261112_063451

Compares old (stale) vs new (fixed-extractor) Qwen AutoGen numbers.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


def extract_inner_output(record: dict) -> str:
    """Extract the inner answer string from a trial record's output field.

    The output field is often a stringified Python dictionary like:
      "{'output': 'Paris', 'latency_ms': 1234, ...}"
    Returns the inner 'output' value, or the raw string if not a dict.
    """
    raw = record.get("output", "")
    if not isinstance(raw, str):
        raw = str(raw)
    if raw.startswith("{"):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict) and "output" in parsed:
                return str(parsed["output"])
        except (ValueError, SyntaxError):
            pass
    return raw


def load_trials(batch_dir: Path) -> list[dict]:
    """Load all trial records from a batch directory."""
    trials = []
    for f in sorted(batch_dir.glob("*.json")):
        if f.name in ("combined_summary.json", "experiment_metadata.json",
                       "manifest.json", "checkpoint.json"):
            continue
        with open(f) as fh:
            data = json.load(fh)
        if isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict) and "framework" in rec:
                    trials.append(rec)
    return trials


def analyze(trials: list[dict]) -> None:
    """Print AutoGen-specific analysis for Table 7 update."""
    executed = [t for t in trials if t.get("status", "completed") == "completed"]
    print(f"Total records: {len(trials)}, Executed: {len(executed)}")

    # Group by condition
    conditions: dict[str, list[dict]] = {}
    for t in executed:
        key = t.get("failure_type", "unknown")
        conditions.setdefault(key, []).append(t)

    for cond_name in ["none", "timeout", "malformed_output", "rate_limit", "api_error"]:
        runs = conditions.get(cond_name, [])
        if not runs:
            print(f"\n  {cond_name}: NO DATA")
            continue

        n = len(runs)
        n_success = sum(1 for r in runs if r.get("success", False))
        n_fail = n - n_success

        # Correctness on assessable prompts (indices 0 and 2)
        assessable = [r for r in runs if r.get("prompt_index", -1) in (0, 2)]
        n_assess = len(assessable)
        n_correct = sum(1 for r in assessable if r.get("grade_correct", False))

        # Empty output check (parse inner answer from serialized dict)
        n_empty = sum(1 for r in runs if not extract_inner_output(r).strip())

        # Tool counts (from first record)
        tool_calls = runs[0].get("injector_calls", 0) if runs else 0
        injections = runs[0].get("injector_injections", 0) if runs else 0

        # Post-retry correctness
        n_retry_correct = 0
        for r in assessable:
            if r.get("grade_correct", False):
                n_retry_correct += 1
            elif r.get("recovery_output", ""):
                out = str(r["recovery_output"]).lower()
                if r.get("prompt_index") == 0 and "paris" in out:
                    n_retry_correct += 1
                elif r.get("prompt_index") == 2 and "127.05" in out:
                    n_retry_correct += 1

        print(f"\n  {cond_name}:")
        print(f"    Runs: {n}, Failures: {n_fail}, Success: {n_success}")
        print(f"    Assessable: {n_assess}, Correct: {n_correct}/{n_assess} ({n_correct/n_assess*100:.1f}%)" if n_assess else "    No assessable")
        print(f"    Post-retry correct: {n_retry_correct}/{n_assess}" if n_assess else "")
        print(f"    Empty outputs: {n_empty}/{n}")
        print(f"    Tool calls: {tool_calls}, Injections: {injections}")

    # Summary for Table 7
    all_assess_baseline = [r for r in conditions.get("none", []) if r.get("prompt_index", -1) in (0, 2)]
    all_assess_injection = []
    for cond in ["timeout", "malformed_output", "rate_limit", "api_error"]:
        all_assess_injection.extend(r for r in conditions.get(cond, []) if r.get("prompt_index", -1) in (0, 2))

    b_correct = sum(1 for r in all_assess_baseline if r.get("grade_correct", False))
    i_correct = sum(1 for r in all_assess_injection if r.get("grade_correct", False))
    b_total = len(all_assess_baseline)
    i_total = len(all_assess_injection)

    print(f"\n{'='*60}")
    print(f"TABLE 7 UPDATE — Qwen AutoGen (new extractor)")
    print(f"{'='*60}")
    # Compute error rates from data (not hardcoded)
    bl_runs = conditions.get("none", [])
    bl_fail = sum(1 for r in bl_runs if not r.get("success", False))
    inj_runs = []
    for c in ["timeout", "malformed_output", "rate_limit", "api_error"]:
        inj_runs.extend(conditions.get(c, []))
    inj_fail = sum(1 for r in inj_runs if not r.get("success", False))
    inj_n = len(inj_runs)

    # Empty-answer counts
    bl_empty = sum(1 for r in bl_runs if not extract_inner_output(r).strip())
    inj_empty = sum(1 for r in inj_runs if not extract_inner_output(r).strip())

    print(f"  Error rate (baseline): {bl_fail}/{len(bl_runs)} ({bl_fail/len(bl_runs)*100:.1f}%)" if bl_runs else "  No baseline")
    print(f"  Error rate (injection): {inj_fail}/{inj_n} ({inj_fail/inj_n*100:.1f}%)" if inj_n else "  No injection")
    print(f"  Empty answers (baseline): {bl_empty}/{len(bl_runs)}")
    print(f"  Empty answers (injection): {inj_empty}/{inj_n}")
    print(f"  Answer-string presence baseline: {b_correct}/{b_total} ({b_correct/b_total*100:.1f}%)" if b_total else "  No baseline assessable")
    print(f"  Answer-string presence injection: {i_correct}/{i_total} ({i_correct/i_total*100:.1f}%)" if i_total else "  No injection assessable")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <batch_dir>")
        sys.exit(1)
    batch = Path(sys.argv[1])
    if not batch.exists():
        print(f"Batch dir not found: {batch}")
        sys.exit(1)
    trials = load_trials(batch)
    if not trials:
        print("No trial records found")
        sys.exit(1)
    analyze(trials)


if __name__ == "__main__":
    main()
