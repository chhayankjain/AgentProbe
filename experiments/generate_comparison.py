"""Generate cross-framework comparison plots and architecture diagrams.

Reads combined summary data from all three experiment branches
(LangGraph, LangChain, AutoGen) and produces paper-ready figures.

Usage::

    .venv/bin/python experiments/generate_comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PLOTS_DIR = Path(__file__).parent / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
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
FW_COLORS = {
    "langgraph": "#2ecc71",
    "langchain": "#3498db",
    "autogen": "#e74c3c",
}
FW_LABELS = {
    "langgraph": "LangGraph",
    "langchain": "LangChain",
    "autogen": "AutoGen",
}


def load_combined() -> pd.DataFrame:
    # Try local combined file first, else build from tmp
    combined_path = Path(__file__).parent / "results" / "combined_summary.json"
    if not combined_path.exists():
        combined_path = Path("/tmp/combined_summary.json")
    with open(combined_path) as f:
        return pd.DataFrame(json.load(f))


# ── Plot 1: Failure Rate Comparison (grouped by framework) ──

def plot_failure_rate_comparison(df: pd.DataFrame) -> None:
    tasks = list(TASK_LABELS.keys())
    frameworks = list(FW_COLORS.keys())
    failure_types = [ft for ft in FAILURE_LABELS if ft != "none"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, task in zip(axes, tasks):
        x = np.arange(len(failure_types))
        width = 0.25
        for i, fw in enumerate(frameworks):
            rates = []
            for ft in failure_types:
                row = df[(df["framework"] == fw) & (df["task"] == task) & (df["failure_type"] == ft)]
                rates.append(row["failure_rate"].values[0] * 100 if not row.empty else 0)
            bars = ax.bar(x + i * width, rates, width, label=FW_LABELS[fw], color=FW_COLORS[fw], alpha=0.85)
            for bar, rate in zip(bars, rates):
                if rate > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f"{rate:.0f}%", ha="center", va="bottom", fontsize=8)

        ax.set_title(TASK_LABELS[task])
        ax.set_xticks(x + width)
        ax.set_xticklabels([FAILURE_LABELS[ft] for ft in failure_types], rotation=25, ha="right")
        ax.set_ylabel("Failure Rate (%)" if task == tasks[0] else "")
        ax.set_ylim(0, 55)

    axes[0].legend(loc="upper left")
    fig.suptitle("Failure Rate Comparison: LangGraph vs LangChain vs AutoGen", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "comparison_failure_rate.png")
    plt.close(fig)
    print("  Saved: comparison_failure_rate.png")


# ── Plot 2: Recovery Rate Comparison ──

def plot_recovery_rate_comparison(df: pd.DataFrame) -> None:
    tasks = list(TASK_LABELS.keys())
    frameworks = list(FW_COLORS.keys())
    failure_types = [ft for ft in FAILURE_LABELS if ft != "none"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, task in zip(axes, tasks):
        x = np.arange(len(failure_types))
        width = 0.25
        for i, fw in enumerate(frameworks):
            rates = []
            for ft in failure_types:
                row = df[(df["framework"] == fw) & (df["task"] == task) & (df["failure_type"] == ft)]
                rates.append(row["recovery_rate"].values[0] * 100 if not row.empty else 0)
            ax.bar(x + i * width, rates, width, label=FW_LABELS[fw], color=FW_COLORS[fw], alpha=0.85)

        ax.set_title(TASK_LABELS[task])
        ax.set_xticks(x + width)
        ax.set_xticklabels([FAILURE_LABELS[ft] for ft in failure_types], rotation=25, ha="right")
        ax.set_ylabel("Recovery Rate (%)" if task == tasks[0] else "")
        ax.set_ylim(0, 115)

    axes[0].legend(loc="lower left")
    fig.suptitle("Recovery Rate Comparison: LangGraph vs LangChain vs AutoGen", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "comparison_recovery_rate.png")
    plt.close(fig)
    print("  Saved: comparison_recovery_rate.png")


# ── Plot 3: Baseline Latency Comparison (no injection) ──

def plot_baseline_latency(df: pd.DataFrame) -> None:
    baseline = df[df["failure_type"] == "none"].copy()
    tasks = list(TASK_LABELS.keys())
    frameworks = list(FW_COLORS.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tasks))
    width = 0.25

    for i, fw in enumerate(frameworks):
        means = []
        p99s = []
        for task in tasks:
            row = baseline[(baseline["framework"] == fw) & (baseline["task"] == task)]
            means.append(row["mean_latency_ms"].values[0] / 1000 if not row.empty else 0)
            p99s.append(row["p99_latency_ms"].values[0] / 1000 if not row.empty else 0)

        bars = ax.bar(x + i * width, means, width, label=f"{FW_LABELS[fw]} (mean)",
                      color=FW_COLORS[fw], alpha=0.85)
        # Add P99 markers
        ax.scatter(x + i * width, p99s, marker="^", color=FW_COLORS[fw],
                   s=60, zorder=5, edgecolors="black", linewidths=0.5)

        for j, (m, p) in enumerate(zip(means, p99s)):
            ax.text(x[j] + i * width, m + 0.5, f"{m:.1f}s", ha="center", fontsize=8)

    ax.set_xlabel("Task Type")
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Baseline Latency (No Injection): Mean + P99 Markers")
    ax.set_xticks(x + width)
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks])
    ax.legend()

    # Custom legend for P99 marker
    p99_marker = plt.Line2D([], [], marker="^", color="gray", linestyle="None",
                            markersize=8, label="P99")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(p99_marker)
    ax.legend(handles=handles)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "comparison_baseline_latency.png")
    plt.close(fig)
    print("  Saved: comparison_baseline_latency.png")


# ── Plot 4: MTTR Comparison (heatmap) ──

def plot_mttr_heatmap(df: pd.DataFrame) -> None:
    frameworks = list(FW_COLORS.keys())
    failure_types = [ft for ft in FAILURE_LABELS if ft != "none"]
    tasks = list(TASK_LABELS.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=True)

    for ax, task in zip(axes, tasks):
        matrix = np.zeros((len(frameworks), len(failure_types)))
        for i, fw in enumerate(frameworks):
            for j, ft in enumerate(failure_types):
                row = df[(df["framework"] == fw) & (df["task"] == task) & (df["failure_type"] == ft)]
                if not row.empty:
                    matrix[i, j] = row["mttr_ms"].values[0] / 1000

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(22, matrix.max()))
        ax.set_xticks(range(len(failure_types)))
        ax.set_xticklabels([FAILURE_LABELS[ft] for ft in failure_types], rotation=25, ha="right")
        ax.set_yticks(range(len(frameworks)))
        ax.set_yticklabels([FW_LABELS[fw] for fw in frameworks])
        ax.set_title(TASK_LABELS[task])

        for i in range(len(frameworks)):
            for j in range(len(failure_types)):
                val = matrix[i, j]
                color = "white" if val > matrix.max() * 0.5 else "black"
                text = f"{val:.1f}s" if val > 0 else "N/A"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    fig.suptitle("Mean Time to Recovery (MTTR) — Framework Comparison", fontsize=14)
    fig.colorbar(im, ax=axes, label="Seconds", shrink=0.8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "comparison_mttr_heatmap.png")
    plt.close(fig)
    print("  Saved: comparison_mttr_heatmap.png")


# ── Plot 5: Summary Radar Chart ──

def plot_summary_radar(df: pd.DataFrame) -> None:
    """Radar chart comparing frameworks across key metrics (tool_use task)."""
    frameworks = list(FW_COLORS.keys())
    tool_df = df[df["task"] == "tool_use"]

    metrics = {
        "Baseline\nSuccess": [],
        "Avg Recovery\nRate": [],
        "Low Mean\nLatency": [],
        "Low P99\nLatency": [],
        "Low MTTR": [],
    }

    for fw in frameworks:
        fw_data = tool_df[tool_df["framework"] == fw]
        baseline = fw_data[fw_data["failure_type"] == "none"]
        injected = fw_data[fw_data["failure_type"] != "none"]

        # Baseline success (1 - failure_rate), normalized 0-1
        metrics["Baseline\nSuccess"].append(1 - baseline["failure_rate"].values[0] if not baseline.empty else 0)
        # Avg recovery rate across injection types
        metrics["Avg Recovery\nRate"].append(injected["recovery_rate"].mean() if not injected.empty else 0)
        # Inverse of mean latency (normalized: lower is better → higher score)
        mean_lat = baseline["mean_latency_ms"].values[0] if not baseline.empty else 20000
        metrics["Low Mean\nLatency"].append(max(0, 1 - mean_lat / 20000))
        # Inverse of P99
        p99 = baseline["p99_latency_ms"].values[0] if not baseline.empty else 20000
        metrics["Low P99\nLatency"].append(max(0, 1 - p99 / 25000))
        # Inverse of avg MTTR
        avg_mttr = injected["mttr_ms"].mean() if not injected.empty else 0
        metrics["Low MTTR"].append(max(0, 1 - avg_mttr / 20000))

    labels = list(metrics.keys())
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for fw in frameworks:
        idx = frameworks.index(fw)
        values = [metrics[k][idx] for k in labels]
        values += values[:1]
        ax.fill(angles, values, alpha=0.15, color=FW_COLORS[fw])
        ax.plot(angles, values, "o-", linewidth=2, label=FW_LABELS[fw], color=FW_COLORS[fw])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_title("Framework Reliability Profile (Tool Use Task)", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "comparison_radar.png")
    plt.close(fig)
    print("  Saved: comparison_radar.png")


# ── Architecture Diagram ──

def draw_architecture_diagram() -> None:
    """Draw AgentProbe system architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("AgentProbe — System Architecture", fontsize=16, fontweight="bold", pad=15)

    def box(x, y, w, h, label, color, sublabel=None, fontsize=11):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                        facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.12 if sublabel else 0), label,
                ha="center", va="center", fontsize=fontsize, fontweight="bold")
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.18, sublabel,
                    ha="center", va="center", fontsize=8, style="italic", color="#333")

    def arrow(x1, y1, x2, y2, label="", color="black"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.15, label, ha="center", va="bottom", fontsize=8, color=color)

    # --- Layer 1: Agent Frameworks (top) ---
    box(0.5, 7.2, 3.5, 1.2, "LangGraph Agent", "#2ecc71", "ReAct + StateGraph")
    box(5.25, 7.2, 3.5, 1.2, "LangChain Agent", "#3498db", "AgentExecutor")
    box(10, 7.2, 3.5, 1.2, "AutoGen Agent", "#e74c3c", "UserProxy + Assistant")

    # --- Layer 2: Injector (middle-left) ---
    box(0.5, 4.8, 5.5, 1.8, "Failure Injector", "#e67e22")
    # Sub-items inside injector
    for i, label in enumerate(["Tool Failure", "Orchestration", "Context", "Latency"]):
        ax.text(1.2 + i * 1.35, 5.2, label, ha="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#fdebd0", edgecolor="#e67e22", linewidth=0.8))

    # --- Layer 2: Observer (middle-right) ---
    box(7, 4.8, 6.5, 1.8, "Observer Layer", "#9b59b6")
    for i, label in enumerate(["OTel Tracer", "Classifier", "Prometheus"]):
        ax.text(8.2 + i * 2.1, 5.2, label, ha="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ebdef0", edgecolor="#9b59b6", linewidth=0.8))

    # --- Layer 3: Benchmark Engine (bottom-center) ---
    box(3, 2.2, 8, 1.8, "Benchmark Runner + Metrics", "#1abc9c")
    for i, label in enumerate(["Task Generator", "RunResult", "MetricsCalc", "CSV/JSON Export"]):
        ax.text(4.0 + i * 2.0, 2.6, label, ha="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#d1f2eb", edgecolor="#1abc9c", linewidth=0.8))

    # --- Layer 4: Output (bottom) ---
    box(1, 0.3, 3.5, 1.2, "Results & CSVs", "#bdc3c7", "experiments/results/")
    box(5.5, 0.3, 3.5, 1.2, "Plots & Figures", "#bdc3c7", "generate_plots.py")
    box(10, 0.3, 3.5, 1.2, "Paper (LaTeX)", "#bdc3c7", "paper/main.tex")

    # --- Arrows ---
    # Agents → Injector
    arrow(2.25, 7.2, 3.25, 6.6, "wraps tools", "#e67e22")
    arrow(7.0, 7.2, 4.5, 6.6, "", "#e67e22")
    # Agents → Observer
    arrow(7.0, 7.2, 10.25, 6.6, "traces spans", "#9b59b6")
    arrow(2.25, 7.2, 8.5, 6.6, "", "#9b59b6")
    # Injector → Benchmark
    arrow(3.25, 4.8, 5.5, 4.0, "failure configs", "#1abc9c")
    # Observer → Benchmark
    arrow(10.25, 4.8, 8.5, 4.0, "classifications", "#1abc9c")
    # Benchmark → Outputs
    arrow(5.0, 2.2, 2.75, 1.5, "", "gray")
    arrow(7.0, 2.2, 7.25, 1.5, "", "gray")
    arrow(9.0, 2.2, 11.75, 1.5, "", "gray")

    # LLM Backend label
    ax.text(7, 8.8, "LLM Backend: Ollama (llama3.1) / OpenAI / Anthropic",
            ha="center", fontsize=10, style="italic", color="#666",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9f9f9", edgecolor="#ccc"))

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "architecture_diagram.png")
    plt.close(fig)
    print("  Saved: architecture_diagram.png")


# ── Experiment Flowchart ──

def draw_experiment_flowchart() -> None:
    """Draw the experiment execution flowchart."""
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_title("AgentProbe — Experiment Flowchart", fontsize=16, fontweight="bold", pad=15)

    def box(x, y, w, h, label, color, sublabel=None, shape="round"):
        if shape == "diamond":
            diamond = plt.Polygon([(x + w/2, y + h), (x + w, y + h/2),
                                    (x + w/2, y), (x, y + h/2)],
                                   facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.85)
            ax.add_patch(diamond)
            ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=9, fontweight="bold")
        else:
            rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                            facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.85)
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2 + (0.1 if sublabel else 0), label,
                    ha="center", va="center", fontsize=10, fontweight="bold")
            if sublabel:
                ax.text(x + w/2, y + h/2 - 0.15, sublabel,
                        ha="center", va="center", fontsize=7.5, style="italic", color="#333")

    def arrow(x1, y1, x2, y2, label="", color="black"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my, label, fontsize=8, color=color, va="center")

    # Step 1: Start
    box(4, 12.8, 4, 0.7, "Start: Configure Experiment", "#ecf0f1")
    arrow(6, 12.8, 6, 12.3)

    # Step 2: Select framework
    box(3.5, 11.5, 5, 0.7, "Select Framework", "#3498db",
        "LangGraph / LangChain / AutoGen")
    arrow(6, 11.5, 6, 11.0)

    # Step 3: Select task
    box(3.5, 10.2, 5, 0.7, "Select Task Type", "#2ecc71",
        "tool_use / rag / multi_agent")
    arrow(6, 10.2, 6, 9.7)

    # Step 4: Select failure type
    box(3.5, 8.9, 5, 0.7, "Select Failure Injection", "#e67e22",
        "none / timeout / malformed / rate_limit / api_error")
    arrow(6, 8.9, 6, 8.4)

    # Step 5: Build agent + injector
    box(3.5, 7.6, 5, 0.7, "Build Agent + Inject Failures", "#9b59b6",
        "LLM + Tools + ToolFailureInjector")
    arrow(6, 7.6, 6, 7.1)

    # Step 6: Run N trials loop
    box(3.5, 6.3, 5, 0.7, "Run N=10 Trials (BenchmarkRunner)", "#1abc9c",
        "Per trial: invoke agent → record RunResult")
    arrow(6, 6.3, 6, 5.8)

    # Step 7: Classify failures
    box(3.5, 5.0, 5, 0.7, "Classify Failures (Observer)", "#9b59b6",
        "FailureClassifier → category + severity")
    arrow(6, 5.0, 6, 4.5)

    # Step 8: Attempt recovery
    box(3.5, 3.7, 5, 0.7, "Attempt Recovery (1 retry)", "#e74c3c",
        "Re-invoke agent → measure MTTR")
    arrow(6, 3.7, 6, 3.2)

    # Step 9: Compute metrics
    box(3.5, 2.4, 5, 0.7, "Compute Metrics", "#1abc9c",
        "failure_rate, recovery_rate, MTTR, P50/P90/P99")
    arrow(6, 2.4, 6, 1.9)

    # Step 10: Save results
    box(3.5, 1.1, 5, 0.7, "Save CSV + JSON + Plots", "#bdc3c7",
        "experiments/results/{fw}_{task}_{failure}.*")
    arrow(6, 1.1, 6, 0.6)

    # End
    box(4, 0.0, 4, 0.5, "Done — Next Config", "#ecf0f1")

    # Loop annotation
    ax.annotate("", xy=(9.2, 12.3), xytext=(9.2, 0.25),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.5, linestyle="--"))
    ax.text(9.5, 6.5, "Repeat for each\n(framework × task ×\nfailure_type)\ncombination\n\n3 × 3 × 5 = 45 configs\n450 total runs",
            fontsize=9, color="#666", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9f9f9", edgecolor="#ccc"))

    # Experiment matrix summary
    ax.text(1.5, 6.5, "Experiment Matrix:\n\nFrameworks:\n  • LangGraph\n  • LangChain\n  • AutoGen\n\n"
            "Tasks:\n  • Tool Use\n  • RAG\n  • Multi-Agent\n\n"
            "Failures:\n  • Baseline\n  • Timeout\n  • Malformed\n  • Rate Limit\n  • API Error",
            fontsize=8, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf2f8", edgecolor="#3498db", linewidth=1))

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "experiment_flowchart.png")
    plt.close(fig)
    print("  Saved: experiment_flowchart.png")


# ── Main ──

def main() -> None:
    print("Loading combined results...")
    df = load_combined()
    print(f"  {len(df)} configs across {df['framework'].nunique()} frameworks")

    # Save combined summary into results/
    combined_path = Path(__file__).parent / "results" / "combined_summary.json"
    df.to_json(combined_path, orient="records", indent=2)
    print(f"  Combined summary saved to {combined_path}")

    print("\nGenerating comparison plots...")
    plot_failure_rate_comparison(df)
    plot_recovery_rate_comparison(df)
    plot_baseline_latency(df)
    plot_mttr_heatmap(df)
    plot_summary_radar(df)

    print("\nGenerating diagrams...")
    draw_architecture_diagram()
    draw_experiment_flowchart()

    print(f"\nAll outputs saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
