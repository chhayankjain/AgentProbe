"""Generate paper-ready plots from AgentProbe benchmark results.

Reads experiments/results/summary.json and per-config CSV files,
produces PNG figures in experiments/results/plots/.

Usage::

    .venv/bin/python experiments/generate_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Paper-friendly style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
})

TASK_LABELS = {"tool_use": "Tool Use", "rag": "RAG", "multi_agent": "Multi-Agent"}
FAILURE_LABELS = {
    "none": "Baseline",
    "timeout": "Timeout",
    "malformed_output": "Malformed",
    "rate_limit": "Rate Limit",
    "api_error": "API Error",
}
COLORS = {
    "none": "#2ecc71",
    "timeout": "#e74c3c",
    "malformed_output": "#e67e22",
    "rate_limit": "#3498db",
    "api_error": "#9b59b6",
}


def load_summary() -> pd.DataFrame:
    with open(RESULTS_DIR / "summary.json") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def load_all_runs() -> pd.DataFrame:
    """Load all per-run CSV files into a single DataFrame."""
    frames = []
    for csv_path in sorted(RESULTS_DIR.glob("langgraph_*.csv")):
        df = pd.read_csv(csv_path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Plot 1: Failure Rate by Task and Failure Type (grouped bar) ──


def plot_failure_rate(df: pd.DataFrame) -> None:
    tasks = list(TASK_LABELS.keys())
    failure_types = list(FAILURE_LABELS.keys())

    x = np.arange(len(tasks))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, ft in enumerate(failure_types):
        rates = [
            df[(df["task"] == t) & (df["failure_type"] == ft)]["failure_rate"].values[0] * 100
            for t in tasks
        ]
        bars = ax.bar(x + i * width, rates, width, label=FAILURE_LABELS[ft], color=COLORS[ft])
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{rate:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Task Type")
    ax.set_ylabel("Failure Rate (%)")
    ax.set_title("Failure Rate by Task Type and Injection Mode (LangGraph)")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks])
    ax.set_ylim(0, 55)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "failure_rate_by_task.png")
    plt.close(fig)
    print(f"  Saved: failure_rate_by_task.png")


# ── Plot 2: Recovery Rate (grouped bar) ──


def plot_recovery_rate(df: pd.DataFrame) -> None:
    tasks = list(TASK_LABELS.keys())
    failure_types = [ft for ft in FAILURE_LABELS if ft != "none"]

    x = np.arange(len(tasks))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, ft in enumerate(failure_types):
        rates = [
            df[(df["task"] == t) & (df["failure_type"] == ft)]["recovery_rate"].values[0] * 100
            for t in tasks
        ]
        ax.bar(x + i * width, rates, width, label=FAILURE_LABELS[ft], color=COLORS[ft])

    ax.set_xlabel("Task Type")
    ax.set_ylabel("Recovery Rate (%)")
    ax.set_title("Recovery Rate After Failure Injection (LangGraph)")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks])
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "recovery_rate_by_task.png")
    plt.close(fig)
    print(f"  Saved: recovery_rate_by_task.png")


# ── Plot 3: Latency Distribution (box plot from per-run data) ──


def plot_latency_boxplot(runs: pd.DataFrame) -> None:
    if runs.empty:
        print("  Skipped: latency_boxplot.png (no per-run data)")
        return

    tasks = list(TASK_LABELS.keys())
    failure_types = list(FAILURE_LABELS.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, task in zip(axes, tasks):
        data = []
        labels = []
        colors = []
        for ft in failure_types:
            subset = runs[(runs["task"] == task) & (runs["failure_type"] == ft)]
            if not subset.empty:
                data.append(subset["latency_ms"].values / 1000)  # convert to seconds
                labels.append(FAILURE_LABELS[ft])
                colors.append(COLORS[ft])

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(TASK_LABELS[task])
        ax.set_ylabel("Latency (s)" if task == tasks[0] else "")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Latency Distribution by Task and Failure Type (LangGraph)", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "latency_boxplot.png")
    plt.close(fig)
    print(f"  Saved: latency_boxplot.png")


# ── Plot 4: MTTR Comparison (horizontal bar) ──


def plot_mttr(df: pd.DataFrame) -> None:
    # Filter only rows with actual failures
    df_fail = df[df["failure_type"] != "none"].copy()
    df_fail["label"] = df_fail.apply(
        lambda r: f"{TASK_LABELS[r['task']]} / {FAILURE_LABELS[r['failure_type']]}", axis=1
    )
    df_fail = df_fail.sort_values("mttr_ms", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [COLORS[ft] for ft in df_fail["failure_type"]]
    ax.barh(df_fail["label"], df_fail["mttr_ms"] / 1000, color=colors, height=0.6)
    ax.set_xlabel("Mean Time to Recovery (s)")
    ax.set_title("MTTR Across Task Types and Failure Modes (LangGraph)")

    for i, (_, row) in enumerate(df_fail.iterrows()):
        ax.text(row["mttr_ms"] / 1000 + 0.2, i, f"{row['mttr_ms']/1000:.1f}s",
                va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mttr_comparison.png")
    plt.close(fig)
    print(f"  Saved: mttr_comparison.png")


# ── Plot 5: P99 Latency Heatmap ──


def plot_p99_heatmap(df: pd.DataFrame) -> None:
    tasks = list(TASK_LABELS.keys())
    failure_types = list(FAILURE_LABELS.keys())

    matrix = np.zeros((len(tasks), len(failure_types)))
    for i, t in enumerate(tasks):
        for j, ft in enumerate(failure_types):
            row = df[(df["task"] == t) & (df["failure_type"] == ft)]
            if not row.empty:
                matrix[i, j] = row["p99_latency_ms"].values[0] / 1000

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(failure_types)))
    ax.set_xticklabels([FAILURE_LABELS[ft] for ft in failure_types])
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([TASK_LABELS[t] for t in tasks])

    for i in range(len(tasks)):
        for j in range(len(failure_types)):
            ax.text(j, i, f"{matrix[i, j]:.1f}s", ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() * 0.6 else "black",
                    fontsize=10, fontweight="bold")

    ax.set_title("P99 Latency Heatmap (seconds) — LangGraph")
    fig.colorbar(im, ax=ax, label="Seconds")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p99_heatmap.png")
    plt.close(fig)
    print(f"  Saved: p99_heatmap.png")


# ── Main ──


def main() -> None:
    print("Loading results...")
    summary = load_summary()
    runs = load_all_runs()
    print(f"  Summary: {len(summary)} configs, Per-run: {len(runs)} rows")

    print("\nGenerating plots...")
    plot_failure_rate(summary)
    plot_recovery_rate(summary)
    plot_latency_boxplot(runs)
    plot_mttr(summary)
    plot_p99_heatmap(summary)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
