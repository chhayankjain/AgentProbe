"""Generate paper-ready plots from AgentProbe benchmark results.

Produces:
  1. failure_rate_by_framework.png — Grouped bar chart of failure rates
  2. recovery_rate_heatmap.png    — Heatmap of recovery rates by framework × failure type
  3. latency_comparison.png       — Box plot of P99 latencies across frameworks
  4. failure_rate_by_task.png     — Failure rates broken down by task type
  5. summary_dashboard.png        — 2×2 panel combining key results

Usage::

    .venv/bin/python experiments/generate_plots.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"


def load_results() -> list[dict]:
    path = RESULTS_DIR / "combined_summary.json"
    if not path.exists():
        print(f"No combined_summary.json found at {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def plot_failure_rate_by_framework(results: list[dict]) -> None:
    """Grouped bar chart: failure rate by framework × failure type."""
    frameworks = sorted(set(r["framework"] for r in results))
    failure_types = [ft for ft in ["none", "timeout", "malformed_output", "rate_limit", "api_error"]
                     if any(r["failure_type"] == ft for r in results)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(failure_types))
    width = 0.25
    colors = {"langgraph": "#2196F3", "langchain": "#FF9800", "autogen": "#4CAF50"}

    for i, fw in enumerate(frameworks):
        rates = []
        for ft in failure_types:
            matching = [r for r in results if r["framework"] == fw and r["failure_type"] == ft]
            if matching:
                total_failures = sum(r["n_failures"] for r in matching)
                total_runs = sum(r["n_runs"] for r in matching)
                rates.append(total_failures / total_runs * 100 if total_runs else 0)
            else:
                rates.append(0)
        ax.bar(x + i * width, rates, width, label=fw.capitalize(),
               color=colors.get(fw, "#999"), edgecolor="white")

    ax.set_xlabel("Failure Injection Type", fontsize=12)
    ax.set_ylabel("Failure Rate (%)", fontsize=12)
    ax.set_title("Failure Rate by Framework and Injection Type", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([ft.replace("_", " ").title() for ft in failure_types], fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(30, ax.get_ylim()[1] * 1.1))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "failure_rate_by_framework.png", dpi=300)
    plt.close(fig)
    print("  Saved failure_rate_by_framework.png")


def plot_recovery_rate_heatmap(results: list[dict]) -> None:
    """Heatmap of recovery rates."""
    frameworks = sorted(set(r["framework"] for r in results))
    failure_types = [ft for ft in ["timeout", "malformed_output", "rate_limit", "api_error"]
                     if any(r["failure_type"] == ft for r in results)]

    data = np.zeros((len(frameworks), len(failure_types)))
    for i, fw in enumerate(frameworks):
        for j, ft in enumerate(failure_types):
            matching = [r for r in results if r["framework"] == fw and r["failure_type"] == ft]
            if matching:
                total_failures = sum(r["n_failures"] for r in matching)
                if total_failures > 0:
                    # Approximate recovered count from recovery_rate × n_failures
                    total_recovered = sum(
                        int(r["recovery_rate"] * r["n_failures"]) for r in matching
                    )
                    data[i, j] = total_recovered / total_failures * 100
                else:
                    data[i, j] = float("nan")

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(failure_types)))
    ax.set_xticklabels([ft.replace("_", " ").title() for ft in failure_types], fontsize=10)
    ax.set_yticks(range(len(frameworks)))
    ax.set_yticklabels([fw.capitalize() for fw in frameworks], fontsize=11)

    # Annotate cells
    for i in range(len(frameworks)):
        for j in range(len(failure_types)):
            val = data[i, j]
            if np.isnan(val):
                text = "N/A\n(0%\nfail)"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color="gray")
            else:
                color = "white" if val < 50 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color)

    ax.set_title("Recovery Rate by Framework and Failure Type", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Recovery Rate (%)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "recovery_rate_heatmap.png", dpi=300)
    plt.close(fig)
    print("  Saved recovery_rate_heatmap.png")


def plot_latency_comparison(results: list[dict]) -> None:
    """Box plot of P99 latencies by framework."""
    frameworks = sorted(set(r["framework"] for r in results))
    colors = {"langgraph": "#2196F3", "langchain": "#FF9800", "autogen": "#4CAF50"}

    fig, ax = plt.subplots(figsize=(8, 5))

    data_by_fw = []
    labels = []
    for fw in frameworks:
        p99s = [r["p99_latency_ms"] for r in results if r["framework"] == fw and r["p99_latency_ms"] > 0]
        if p99s:
            data_by_fw.append(p99s)
            labels.append(fw.capitalize())

    bp = ax.boxplot(data_by_fw, tick_labels=labels, patch_artist=True, widths=0.5)
    for i, fw in enumerate(frameworks):
        if i < len(bp["boxes"]):
            bp["boxes"][i].set_facecolor(colors.get(fw, "#999"))
            bp["boxes"][i].set_alpha(0.7)

    ax.set_ylabel("P99 Latency (ms)", fontsize=12)
    ax.set_title("P99 Latency Distribution by Framework", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "latency_comparison.png", dpi=300)
    plt.close(fig)
    print("  Saved latency_comparison.png")


def plot_failure_rate_by_task(results: list[dict]) -> None:
    """Failure rates broken down by task type."""
    frameworks = sorted(set(r["framework"] for r in results))
    tasks = sorted(set(r["task"] for r in results))
    colors = {"langgraph": "#2196F3", "langchain": "#FF9800", "autogen": "#4CAF50"}

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        task_results = [r for r in results if r["task"] == task and r["failure_type"] != "none"]
        fw_rates = {}
        for fw in frameworks:
            matching = [r for r in task_results if r["framework"] == fw]
            if matching:
                total_f = sum(r["n_failures"] for r in matching)
                total_n = sum(r["n_runs"] for r in matching)
                fw_rates[fw] = total_f / total_n * 100 if total_n else 0
            else:
                fw_rates[fw] = 0

        bars = ax.bar(
            [fw.capitalize() for fw in frameworks],
            [fw_rates[fw] for fw in frameworks],
            color=[colors.get(fw, "#999") for fw in frameworks],
            edgecolor="white",
            width=0.6,
        )
        ax.set_title(task.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_ylim(0, 30)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, fw in zip(bars, frameworks):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                    f"{height:.1f}%", ha="center", va="bottom", fontsize=10)

    axes[0].set_ylabel("Failure Rate (%)", fontsize=12)
    fig.suptitle("Failure Rate by Task Type (Injection Conditions Only)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "failure_rate_by_task.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved failure_rate_by_task.png")


def plot_summary_dashboard(results: list[dict]) -> None:
    """2×2 summary dashboard."""
    frameworks = sorted(set(r["framework"] for r in results))
    colors_list = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Overall failure rate
    ax = axes[0, 0]
    overall_rates = []
    for fw in frameworks:
        matching = [r for r in results if r["framework"] == fw]
        total_f = sum(r["n_failures"] for r in matching)
        total_n = sum(r["n_runs"] for r in matching)
        overall_rates.append(total_f / total_n * 100 if total_n else 0)

    bars = ax.bar([fw.capitalize() for fw in frameworks], overall_rates,
                  color=colors_list[:len(frameworks)], edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.3,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Failure Rate (%)")
    ax.set_title("(a) Overall Failure Rate", fontweight="bold")
    ax.set_ylim(0, max(20, max(overall_rates) * 1.3))
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Recovery rate
    ax = axes[0, 1]
    recovery_rates = []
    for fw in frameworks:
        matching = [r for r in results if r["framework"] == fw and r["failure_type"] != "none"]
        total_f = sum(r["n_failures"] for r in matching)
        total_recovered = sum(int(r["recovery_rate"] * r["n_failures"]) for r in matching)
        recovery_rates.append(total_recovered / total_f * 100 if total_f else 0)

    bars = ax.bar([fw.capitalize() for fw in frameworks], recovery_rates,
                  color=colors_list[:len(frameworks)], edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        label = f"{h:.0f}%" if h > 0 else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2., max(h, 2) + 1,
                label, ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Recovery Rate (%)")
    ax.set_title("(b) Recovery Rate (Injection Conditions)", fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Mean latency (baseline only — injection latency is not comparable
    # for AutoGen because it short-circuits without LLM inference)
    ax = axes[1, 0]
    mean_latencies = []
    for fw in frameworks:
        matching = [r for r in results if r["framework"] == fw and r["failure_type"] == "none"]
        if matching:
            mean_latencies.append(np.mean([r["mean_latency_ms"] for r in matching]))
        else:
            mean_latencies.append(0)

    bars = ax.bar([fw.capitalize() for fw in frameworks], mean_latencies,
                  color=colors_list[:len(frameworks)], edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 300,
                f"{h/1000:.1f}s", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title("(c) Mean Latency (Baseline, No Injection)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: P99 latency by task (baseline only)
    ax = axes[1, 1]
    tasks = sorted(set(r["task"] for r in results))
    x = np.arange(len(tasks))
    width = 0.25
    for i, fw in enumerate(frameworks):
        p99s = []
        for task in tasks:
            matching = [r for r in results if r["framework"] == fw and r["task"] == task and r["failure_type"] == "none"]
            if matching:
                p99s.append(np.mean([r["p99_latency_ms"] for r in matching]))
            else:
                p99s.append(0)
        ax.bar(x + i * width, p99s, width, label=fw.capitalize(),
               color=colors_list[i], edgecolor="white")
    ax.set_ylabel("P99 Latency (ms)")
    ax.set_title("(d) P99 Latency by Task (Baseline)", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([t.replace("_", " ").title() for t in tasks])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("AgentProbe: Cross-Framework Reliability Comparison\n"
                 "(llama3.1, Ollama, 1050 runs)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PLOTS_DIR / "summary_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved summary_dashboard.png")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    print(f"Loaded {len(results)} conditions from {len(set(r['framework'] for r in results))} frameworks")
    print("Generating plots...")

    plot_failure_rate_by_framework(results)
    plot_recovery_rate_heatmap(results)
    plot_latency_comparison(results)
    plot_failure_rate_by_task(results)
    plot_summary_dashboard(results)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
