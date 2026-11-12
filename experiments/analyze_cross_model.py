"""Analyze cross-model Phase 2 results for the paper.

Usage:
    .venv/bin/python experiments/analyze_cross_model.py \
        --batches experiments/results_v2/20261111_113514 \
                  experiments/results_v2/20261111_233435 \
                  experiments/results_v2/20261112_040156 \
        --models llama3.1 mistral:7b qwen2.5:7b
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import ast

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_inner_output(record: dict) -> str:
    """Extract the inner answer string from a trial record's output field.

    The output field is often a stringified Python dictionary like:
      "{'output': 'Paris', 'latency_ms': 1234, ...}"
    This function returns the inner 'output' value ('Paris'), or the
    raw string if it's not a serialized dict.
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


# Prompt indices with verifiable ground truth (calculator questions)
ASSESSABLE_PROMPTS = {2}  # "Calculate 15% of 847" → 127.05
# Prompt 0 ("capital of France") uses keyword grading only in the saved data;
# treat it as assessable via answer-presence check below.
ASSESSABLE_PROMPTS_ANSWER_PRESENCE = {0, 2}

# Expected substrings for answer-presence grading on assessable prompts
EXPECTED_ANSWERS = {
    0: "paris",       # "What is the capital of France?"
    2: "127.05",      # "Calculate 15% of 847"
}

# Files to exclude when globbing batch directories
EXCLUDE_FILES = {
    "experiment_metadata.json", "checkpoint.json", "batch_summary.json",
    "combined_summary.json", "manifest.json", "audit_derived_analysis.json",
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def load_batch(batch_dir: Path) -> list[dict]:
    """Load trial-level records from a batch directory.

    Excludes summary/metadata files and validates required fields.
    """
    records = []
    for f in sorted(batch_dir.glob("*.json")):
        if f.name in EXCLUDE_FILES:
            continue
        data = json.loads(f.read_text())
        if isinstance(data, list):
            for r in data:
                # Validate required fields exist
                if "success" not in r or "framework" not in r:
                    print(f"  WARNING: {f.name} record missing required fields, skipping")
                    continue
                # Skip placeholder/skipped rows
                if r.get("status") == "skipped":
                    continue
                records.append(r)
        elif isinstance(data, dict) and "runs" in data:
            for r in data["runs"]:
                if "success" not in r or "framework" not in r:
                    continue
                if r.get("status") == "skipped":
                    continue
                records.append(r)
    return records


def is_assessable(record: dict) -> bool:
    """Check if a record is for an assessable prompt."""
    return record.get("prompt_index") in ASSESSABLE_PROMPTS_ANSWER_PRESENCE


def check_answer_present(record: dict) -> bool:
    """Check if the expected answer string is present in the inner output."""
    prompt_idx = record.get("prompt_index")
    expected = EXPECTED_ANSWERS.get(prompt_idx)
    if expected is None:
        return False
    output = extract_inner_output(record).lower()
    return expected.lower() in output


def check_answer_present_with_retry(record: dict) -> bool:
    """Check if expected answer string is present in output or recovery output."""
    if check_answer_present(record):
        return True
    prompt_idx = record.get("prompt_index")
    expected = EXPECTED_ANSWERS.get(prompt_idx)
    if expected is None:
        return False
    # Recovery output may also be a serialized dict
    raw_recovery = record.get("recovery_output", "")
    if isinstance(raw_recovery, str) and raw_recovery.startswith("{"):
        try:
            parsed = ast.literal_eval(raw_recovery)
            if isinstance(parsed, dict) and "output" in parsed:
                raw_recovery = str(parsed["output"])
        except (ValueError, SyntaxError):
            pass
    return expected.lower() in str(raw_recovery).lower()


def analyze_condition(runs: list[dict]) -> dict:
    """Analyze a single (framework, fault_type) condition."""
    n = len(runs)
    failures = sum(1 for r in runs if not r.get("success", True))
    recovered = sum(1 for r in runs if r.get("recovered", False))
    empty_outputs = sum(1 for r in runs if not extract_inner_output(r).strip())

    # Keyword-based grading (from saved grade_correct)
    kw_pass = sum(1 for r in runs if r.get("grade_correct", False))

    # Answer-presence correctness on assessable subset
    assessable_runs = [r for r in runs if is_assessable(r)]
    n_assessable = len(assessable_runs)
    correct = sum(1 for r in assessable_runs if check_answer_present(r))
    correct_after_retry = sum(
        1 for r in assessable_runs if check_answer_present_with_retry(r)
    )

    # Tool call counts (use the FIRST record's value — they're condition-wide totals)
    tool_calls = runs[0].get("injector_calls", 0) if runs else 0
    injections = runs[0].get("injector_injections", 0) if runs else 0

    error_rate = failures / n if n > 0 else 0
    recovery_rate = recovered / failures if failures > 0 else None

    result = {
        "n": n,
        "failures": failures,
        "error_rate": round(error_rate, 4),
        "recovered": recovered,
        "recovery_rate": round(recovery_rate, 4) if recovery_rate is not None else None,
        "kw_pass": kw_pass,
        "kw_rate": round(kw_pass / n, 4) if n > 0 else 0,
        "assessable": n_assessable,
        "correct": correct,
        "correct_after_retry": correct_after_retry,
        "empty_outputs": empty_outputs,
        "tool_calls": tool_calls,
        "injections": injections,
    }

    # Add Wilson CIs where meaningful
    if n >= 5:
        lo, hi = wilson_ci(failures, n)
        result["error_rate_ci"] = [round(lo, 4), round(hi, 4)]
    if failures > 0:
        lo, hi = wilson_ci(recovered, failures)
        result["recovery_rate_ci"] = [round(lo, 4), round(hi, 4)]

    return result


def analyze_batch(records: list[dict], model_name: str) -> dict:
    """Extract key metrics from a batch of run records."""
    groups = defaultdict(list)
    for r in records:
        fw = r.get("framework", "unknown")
        cond = r.get("failure_type", "unknown")
        groups[(fw, cond)].append(r)

    results = {"model": model_name, "conditions": {}}

    for (fw, cond), runs in sorted(groups.items()):
        key = f"{fw}/{cond}"
        results["conditions"][key] = analyze_condition(runs)

    return results


def print_comparison(all_results: list[dict]) -> None:
    """Print a cross-model comparison table."""
    models = [r["model"] for r in all_results]

    print("\n" + "=" * 80)
    print("CROSS-MODEL PHASE 2 COMPARISON")
    print("=" * 80)

    # Per-framework summary
    for fw in ["langgraph", "langchain", "autogen"]:
        print(f"\n--- {fw.upper()} ---")
        print(f"{'Condition':<20} " + "  ".join(f"{m:<25}" for m in models))
        print("-" * (20 + 27 * len(models)))

        for cond_label, cond_key in [
            ("Baseline", "none"),
            ("Timeout", "timeout"),
            ("Malformed", "malformed_output"),
            ("Rate Limit", "rate_limit"),
            ("API Error", "api_error"),
        ]:
            key = f"{fw}/{cond_key}"
            values = []
            for r in all_results:
                c = r["conditions"].get(key)
                if c is None:
                    values.append("---")
                    continue
                n = c["n"]
                fail = c["failures"]
                rec = c["recovered"]
                correct = c["correct"]
                assess = c["assessable"]
                calls = c["tool_calls"]
                inj = c["injections"]
                s = f"n={n} f={fail} r={rec} c={correct}/{assess} t={calls}/{inj}"
                values.append(s)
            print(f"{cond_label:<20} " + "  ".join(f"{v:<25}" for v in values))

    # Summary table matching paper format
    print("\n" + "=" * 80)
    print("PAPER TABLE FORMAT (per-condition values)")
    print("=" * 80)
    print(f"\n{'Metric':<25} " + "  ".join(f"{m:<20}" for m in models))
    print("-" * (25 + 22 * len(models)))

    for fw in ["langgraph"]:  # LG/LC identical, show one
        # Use first injection condition as representative
        for cond in ["timeout"]:
            key = f"{fw}/{cond}"
            vals_err, vals_rec, vals_corr, vals_retry, vals_calls, vals_inj = (
                [], [], [], [], [], [],
            )
            for r in all_results:
                c = r["conditions"].get(key)
                if c is None:
                    for lst in [vals_err, vals_rec, vals_corr, vals_retry, vals_calls, vals_inj]:
                        lst.append("---")
                    continue
                ci_err = c.get("error_rate_ci", [0, 0])
                vals_err.append(
                    f"{c['error_rate']:.1%} [{ci_err[0]:.1%}-{ci_err[1]:.1%}]"
                )
                if c["recovery_rate"] is not None:
                    ci_rec = c.get("recovery_rate_ci", [0, 0])
                    vals_rec.append(
                        f"{c['recovery_rate']:.1%} [{ci_rec[0]:.1%}-{ci_rec[1]:.1%}]"
                    )
                else:
                    vals_rec.append("N/A")
                vals_corr.append(f"{c['correct']}/{c['assessable']}")
                vals_retry.append(f"{c['correct_after_retry']}/{c['assessable']}")
                vals_calls.append(str(c["tool_calls"]))
                vals_inj.append(str(c["injections"]))

            print(f"{'LG/LC Error Rate':<25} " + "  ".join(f"{v:<20}" for v in vals_err))
            print(f"{'LG/LC Recovery Rate':<25} " + "  ".join(f"{v:<20}" for v in vals_rec))
            print(f"{'LG/LC Correct (init)':<25} " + "  ".join(f"{v:<20}" for v in vals_corr))
            print(f"{'LG/LC Correct (retry)':<25} " + "  ".join(f"{v:<20}" for v in vals_retry))
            print(f"{'LG/LC Tool Calls':<25} " + "  ".join(f"{v:<20}" for v in vals_calls))
            print(f"{'LG/LC Injections':<25} " + "  ".join(f"{v:<20}" for v in vals_inj))

        # Baseline
        key = f"{fw}/none"
        vals_base_corr = []
        for r in all_results:
            c = r["conditions"].get(key)
            if c:
                vals_base_corr.append(f"{c['correct']}/{c['assessable']}")
            else:
                vals_base_corr.append("---")
        print(f"{'LG/LC Baseline Correct':<25} " + "  ".join(f"{v:<20}" for v in vals_base_corr))

    # AutoGen
    print()
    for cond_label, cond_key in [("AG Baseline", "none"), ("AG Timeout", "timeout")]:
        key = f"autogen/{cond_key}"
        vals = []
        for r in all_results:
            c = r["conditions"].get(key)
            if c is None:
                vals.append("---")
                continue
            vals.append(
                f"n={c['n']} err={c['error_rate']:.1%} "
                f"c={c['correct']}/{c['assessable']} "
                f"empty={c['empty_outputs']}"
            )
        print(f"{cond_label:<25} " + "  ".join(f"{v:<20}" for v in vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model Phase 2 analysis")
    parser.add_argument("--batches", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", default=None, help="Save JSON summary")
    args = parser.parse_args()

    if len(args.batches) != len(args.models):
        print("Error: --batches and --models must have the same number of entries")
        sys.exit(1)

    all_results = []
    for batch_path, model in zip(args.batches, args.models):
        batch_dir = Path(batch_path)
        if not batch_dir.exists():
            print(f"Warning: {batch_dir} does not exist, skipping")
            continue
        records = load_batch(batch_dir)
        print(f"Loaded {len(records)} trial records from {batch_dir.name} ({model})")
        result = analyze_batch(records, model)
        all_results.append(result)

    print_comparison(all_results)

    if args.output:
        Path(args.output).write_text(json.dumps(all_results, indent=2) + "\n")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
