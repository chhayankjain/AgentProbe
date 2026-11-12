"""Generate publication-quality figures for the AgentProbe MAKE paper.

Loads Phase 2 batch data directly and computes metrics, then produces:
  1. cross_model_error_recovery.pdf — Error & recovery rates across models (LG/LC)
  2. cross_model_tool_exposure.pdf  — Tool calls & injection counts across models
  3. cross_model_correctness.pdf    — Answer correctness progression (init -> retry)

Usage::

    .venv/bin/python experiments/generate_paper_figures.py

    # Or with explicit batch dirs:
    .venv/bin/python experiments/generate_paper_figures.py \
        --batches experiments/results_v2/20261111_113514 \
                  experiments/results_v2/20261111_233435 \
                  experiments/results_v2/20261112_040156
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.analyze_cross_model import (
    analyze_batch,
    load_batch,
    wilson_ci,
)

# ---------------------------------------------------------------------------
# Default batch directories (Phase 2 data)
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent / "results_v2"
DEFAULT_BATCHES = [
    ("Llama 3.1",   "20261111_113514"),
    ("Mistral 7B",  "20261111_233435"),
    ("Qwen2.5 7B", "20261112_040156"),
]

FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"

# Style
MODEL_COLORS = ["#2196F3", "#FF9800", "#4CAF50"]
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})


def extract_lglc_metrics(all_results: list[dict]) -> dict:
    """Extract per-model LG/LC injection metrics from analyzed batch results.

    Uses the first injection condition (timeout) as representative since
    LG/LC conditions are identical per model (same tool-call counts, same
    injection counts from seed 42).
    """
    models = [r["model"] for r in all_results]
    metrics = {
        "models": models,
        "n": [], "failures": [], "recoveries": [],
        "error_rate": [], "recovery_rate": [],
        "error_ci_lo": [], "error_ci_hi": [],
        "recovery_ci_lo": [], "recovery_ci_hi": [],
        "tool_calls": [], "injections": [],
        "correct_baseline": [], "correct_injection": [],
        "correct_retry": [], "n_assessable": [],
    }

    for r in all_results:
        # Use langgraph/timeout as representative injection condition
        inj = r["conditions"].get("langgraph/timeout")
        bl = r["conditions"].get("langgraph/none")

        if inj is None:
            raise ValueError(f"No langgraph/timeout data for {r['model']}")

        metrics["n"].append(inj["n"])
        metrics["failures"].append(inj["failures"])
        metrics["recoveries"].append(inj["recovered"])
        metrics["error_rate"].append(round(inj["error_rate"] * 100, 1))
        rec_rate = (inj["recovery_rate"] or 0) * 100
        metrics["recovery_rate"].append(round(rec_rate, 1))

        ci_err = inj.get("error_rate_ci", [0, 0])
        metrics["error_ci_lo"].append(round(ci_err[0] * 100, 1))
        metrics["error_ci_hi"].append(round(ci_err[1] * 100, 1))

        ci_rec = inj.get("recovery_rate_ci", [0, 0])
        metrics["recovery_ci_lo"].append(round(ci_rec[0] * 100, 1))
        metrics["recovery_ci_hi"].append(round(ci_rec[1] * 100, 1))

        metrics["tool_calls"].append(inj["tool_calls"])
        metrics["injections"].append(inj["injections"])

        metrics["correct_injection"].append(inj["correct"])
        metrics["correct_retry"].append(inj["correct_after_retry"])
        metrics["n_assessable"].append(inj["assessable"])

        if bl is not None:
            metrics["correct_baseline"].append(bl["correct"])
        else:
            metrics["correct_baseline"].append(0)

    return metrics


def fig1_error_recovery(lglc: dict) -> None:
    """Side-by-side bars: error rate and recovery rate per model (LG/LC)."""
    models = lglc["models"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))
    x = np.arange(len(models))
    width = 0.55

    # Error rate
    bars = ax1.bar(x, lglc["error_rate"], width, color=MODEL_COLORS[:len(models)],
                   edgecolor="white", zorder=3)
    err_lo = [r - lo for r, lo in zip(lglc["error_rate"], lglc["error_ci_lo"])]
    err_hi = [hi - r for r, hi in zip(lglc["error_rate"], lglc["error_ci_hi"])]
    ax1.errorbar(x, lglc["error_rate"], yerr=[err_lo, err_hi],
                 fmt="none", ecolor="black", capsize=4, capthick=1.2, zorder=4)
    for bar, val in zip(bars, lglc["error_rate"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Error Rate (%)")
    ax1.set_title("(a) Fault-Triggered Error Rate", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, 55)
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # Recovery rate
    bars = ax2.bar(x, lglc["recovery_rate"], width, color=MODEL_COLORS[:len(models)],
                   edgecolor="white", zorder=3)
    rec_lo = [r - lo for r, lo in zip(lglc["recovery_rate"], lglc["recovery_ci_lo"])]
    rec_hi = [hi - r for r, hi in zip(lglc["recovery_rate"], lglc["recovery_ci_hi"])]
    ax2.errorbar(x, lglc["recovery_rate"], yerr=[rec_lo, rec_hi],
                 fmt="none", ecolor="black", capsize=4, capthick=1.2, zorder=4)
    for bar, val, n_f, n_r in zip(bars, lglc["recovery_rate"], lglc["failures"], lglc["recoveries"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 f"{val:.1f}%\n({n_r}/{n_f})", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.set_ylabel("Recovery Rate (%)")
    ax2.set_title("(b) Single-Retry Recovery Rate", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0, 110)
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("LangGraph / LangChain: Per-Condition Rates ($n = 30$, $p = 0.20$)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_model_error_recovery.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved cross_model_error_recovery.pdf")


def fig2_tool_exposure(lglc: dict) -> None:
    """Grouped bars: tool calls and injections per model."""
    models = lglc["models"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    x = np.arange(len(models))
    width = 0.32

    bars1 = ax.bar(x - width / 2, lglc["tool_calls"], width, label="Tool calls",
                   color="#90CAF9", edgecolor="black", linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + width / 2, lglc["injections"], width, label="Injections",
                   color="#FF8A65", edgecolor="black", linewidth=0.5, zorder=3)

    for bar, val in zip(bars1, lglc["tool_calls"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, lglc["injections"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Count (per condition)")
    ax.set_title("Tool-Call Exposure by Model (LG/LC, seed 42)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(lglc["tool_calls"]) + 15)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_model_tool_exposure.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved cross_model_tool_exposure.pdf")


def fig3_correctness(lglc: dict) -> None:
    """Grouped bars showing correctness progression: baseline -> injection -> post-retry."""
    models = lglc["models"]
    n_assess = lglc["n_assessable"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(models))
    width = 0.22

    stages = ["Baseline", "Injection (initial)", "Injection (post-retry)"]
    stage_colors = ["#90CAF9", "#FFB74D", "#81C784"]
    data = [
        [c / n * 100 for c, n in zip(lglc["correct_baseline"], n_assess)],
        [c / n * 100 for c, n in zip(lglc["correct_injection"], n_assess)],
        [c / n * 100 for c, n in zip(lglc["correct_retry"], n_assess)],
    ]
    counts = [lglc["correct_baseline"], lglc["correct_injection"], lglc["correct_retry"]]
    hatches = ["", "//", ""]

    for i, (stage, vals, cnts) in enumerate(zip(stages, data, counts)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=stage,
                      color=stage_colors[i], edgecolor="black", linewidth=0.5,
                      hatch=hatches[i], zorder=3)
        for bar, val, cnt, n in zip(bars, vals, cnts, n_assess):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{cnt}/{n}", ha="center", va="bottom", fontsize=7.5)

    ax.set_ylabel("Assessable-Answer Presence (%)")
    n_label = n_assess[0] if len(set(n_assess)) == 1 else "varies"
    ax.set_title(f"Answer-String Presence Progression (LG/LC, $n = {n_label}$ assessable per condition)",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 118)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=8, ncol=3,
              framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_model_correctness.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved cross_model_correctness.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from batch data")
    parser.add_argument("--batches", nargs="+", default=None,
                        help="Batch directories (default: use built-in Phase 2 paths)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model display names (must match --batches count)")
    args = parser.parse_args()

    # Resolve batch dirs and model names
    if args.batches is not None:
        if args.models is None or len(args.models) != len(args.batches):
            print("Error: --models must match --batches count")
            sys.exit(1)
        batch_specs = list(zip(args.models, args.batches))
    else:
        batch_specs = [
            (name, str(RESULTS_DIR / dirname))
            for name, dirname in DEFAULT_BATCHES
        ]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating paper figures to {FIGURES_DIR}/")

    # Load and analyze all batches
    all_results = []
    for model_name, batch_path in batch_specs:
        batch_dir = Path(batch_path)
        if not batch_dir.exists():
            print(f"  WARNING: {batch_dir} not found, skipping {model_name}")
            continue
        records = load_batch(batch_dir)
        print(f"  Loaded {len(records)} records from {batch_dir.name} ({model_name})")
        result = analyze_batch(records, model_name)
        all_results.append(result)

    if not all_results:
        print("No batch data found. Cannot generate figures.")
        sys.exit(1)

    lglc = extract_lglc_metrics(all_results)

    fig1_error_recovery(lglc)
    fig2_tool_exposure(lglc)
    fig3_correctness(lglc)
    print("Done.")


if __name__ == "__main__":
    main()
