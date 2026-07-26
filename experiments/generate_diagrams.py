"""Generate professional architecture diagrams for the AgentProbe paper.

Produces 4 diagrams using graphviz:
  1. System architecture — full AgentProbe pipeline
  2. Experiment flowchart — how benchmarks are executed
  3. Failure taxonomy — 5 categories with subtypes
  4. Framework internals — side-by-side comparison of LangGraph/LangChain/AutoGen

Usage::
    .venv/bin/python experiments/generate_diagrams.py
"""

from __future__ import annotations

import graphviz
from pathlib import Path

OUT = Path(__file__).parent / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)


# ── Shared style constants ──────────────────────────────────────────────

FONT = "Helvetica"
TITLE_FONT = "Helvetica-Bold"

# Color palette
C_BG = "#FAFBFC"
C_AGENT_LG = "#2D7D46"    # LangGraph green
C_AGENT_LC = "#2563EB"    # LangChain blue
C_AGENT_AG = "#DC2626"    # AutoGen red
C_INJECTOR = "#F59E0B"    # Amber
C_OBSERVER = "#7C3AED"    # Purple
C_BENCH = "#0891B2"       # Cyan
C_LLM = "#6B7280"         # Gray
C_OUTPUT = "#374151"       # Dark gray
C_WHITE = "#FFFFFF"
C_LIGHT = "#F3F4F6"


def _save(dot: graphviz.Digraph | graphviz.Graph, name: str) -> None:
    """Render to PNG and clean up .gv source file."""
    path = str(OUT / name)
    dot.render(path, format="png", cleanup=True)
    print(f"  Saved {path}.png")


# ═══════════════════════════════════════════════════════════════════════
# 1. SYSTEM ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════

def architecture_diagram() -> None:
    dot = graphviz.Digraph("architecture", engine="dot")
    dot.attr(
        rankdir="TB", bgcolor=C_BG, pad="0.5", nodesep="0.6", ranksep="0.8",
        label="AgentProbe — System Architecture",
        labelloc="t", fontsize="24", fontname=TITLE_FONT, fontcolor="#111827",
    )
    dot.attr("node", shape="box", style="rounded,filled", fontname=FONT, fontsize="11",
             margin="0.25,0.15", penwidth="1.5")
    dot.attr("edge", fontname=FONT, fontsize="9", color="#9CA3AF", penwidth="1.2")

    # ── LLM Backend ──
    with dot.subgraph(name="cluster_llm") as s:
        s.attr(label="LLM Backend", style="dashed,rounded", color="#D1D5DB",
               fontname=TITLE_FONT, fontsize="12", fontcolor=C_LLM)
        s.node("ollama", "Ollama\n(llama3.1)", fillcolor="#E5E7EB", fontcolor="#374151")
        s.node("openai", "OpenAI\nAPI", fillcolor="#E5E7EB", fontcolor="#374151")
        s.node("anthropic", "Anthropic\nAPI", fillcolor="#E5E7EB", fontcolor="#374151")

    # ── Agent Frameworks ──
    with dot.subgraph(name="cluster_agents") as s:
        s.attr(label="Agent Frameworks (Interchangeable)", style="rounded,filled",
               color="#D1FAE5", fillcolor="#ECFDF5",
               fontname=TITLE_FONT, fontsize="13", fontcolor="#065F46")

        s.node("langgraph", "LangGraph Agent\nReAct + StateGraph",
               fillcolor=C_AGENT_LG, fontcolor=C_WHITE)
        s.node("langchain", "LangChain Agent\nAgentExecutor + Tools",
               fillcolor=C_AGENT_LC, fontcolor=C_WHITE)
        s.node("autogen", "AutoGen Agent\nUserProxy + Assistant",
               fillcolor=C_AGENT_AG, fontcolor=C_WHITE)

    # ── Failure Injector ──
    with dot.subgraph(name="cluster_injector") as s:
        s.attr(label="Failure Injector", style="rounded,filled",
               color="#FDE68A", fillcolor="#FFFBEB",
               fontname=TITLE_FONT, fontsize="13", fontcolor="#92400E")

        s.node("inj_main", "ToolFailureInjector", fillcolor=C_INJECTOR, fontcolor=C_WHITE)
        for name_id, label in [
            ("tf", "Tool Failure\ntimeout · malformed\nrate_limit · api_error"),
            ("of", "Orchestration\ninfinite_loop · wrong_branch\nstate_corruption"),
            ("cf", "Context Failure\noverflow · lost_state\nhallucinated_history"),
            ("lf", "Latency Failure\np99_spike · cascading\ncold_start · jitter"),
        ]:
            s.node(name_id, label, fillcolor="#FEF3C7", fontcolor="#78350F", fontsize="9")

    # ── Observer Layer ──
    with dot.subgraph(name="cluster_observer") as s:
        s.attr(label="Observer Layer", style="rounded,filled",
               color="#DDD6FE", fillcolor="#F5F3FF",
               fontname=TITLE_FONT, fontsize="13", fontcolor="#4C1D95")

        s.node("tracer", "OTel Tracer\nSpans + Attributes", fillcolor=C_OBSERVER, fontcolor=C_WHITE)
        s.node("classifier", "Failure Classifier\n5 categories · severity", fillcolor="#A78BFA", fontcolor=C_WHITE)
        s.node("prometheus", "Prometheus\nDashboard + Alerts", fillcolor="#8B5CF6", fontcolor=C_WHITE)

    # ── Benchmark Engine ──
    with dot.subgraph(name="cluster_bench") as s:
        s.attr(label="Benchmark Engine", style="rounded,filled",
               color="#A5F3FC", fillcolor="#ECFEFF",
               fontname=TITLE_FONT, fontsize="13", fontcolor="#155E75")

        s.node("runner", "BenchmarkRunner\nN trials per config", fillcolor=C_BENCH, fontcolor=C_WHITE)
        s.node("metrics", "MetricsCalculator\nfailure_rate · recovery_rate\nMTTR · P50/P90/P99", fillcolor="#06B6D4", fontcolor=C_WHITE)
        s.node("tasks", "Task Generator\ntool_use · rag · multi_agent", fillcolor="#22D3EE", fontcolor="#164E63")

    # ── Outputs ──
    with dot.subgraph(name="cluster_output") as s:
        s.attr(label="Outputs", style="rounded,filled",
               color="#D1D5DB", fillcolor="#F9FAFB",
               fontname=TITLE_FONT, fontsize="13", fontcolor="#374151")

        s.node("csv", "Results\nCSV + JSON", fillcolor="#E5E7EB", fontcolor="#1F2937")
        s.node("plots", "Plots\nmatplotlib PNGs", fillcolor="#E5E7EB", fontcolor="#1F2937")
        s.node("paper", "Paper\nLaTeX", fillcolor="#E5E7EB", fontcolor="#1F2937")

    # ── Edges ──
    # LLM → Agents
    for llm in ["ollama", "openai", "anthropic"]:
        for agent in ["langgraph", "langchain", "autogen"]:
            dot.edge(llm, agent, style="dashed", color="#D1D5DB", arrowsize="0.6")

    # Agents → Injector (tool calls pass through)
    for agent in ["langgraph", "langchain", "autogen"]:
        dot.edge(agent, "inj_main", label="tool calls", color=C_INJECTOR, fontcolor="#92400E")

    # Injector subtypes
    for sub in ["tf", "of", "cf", "lf"]:
        dot.edge("inj_main", sub, arrowsize="0.5", color="#F59E0B")

    # Agents → Observer (traces/spans)
    for agent in ["langgraph", "langchain", "autogen"]:
        dot.edge(agent, "tracer", label="spans", color=C_OBSERVER, fontcolor="#4C1D95", style="dashed")

    # Observer internal
    dot.edge("tracer", "classifier", color=C_OBSERVER)
    dot.edge("tracer", "prometheus", color=C_OBSERVER)

    # Benchmark orchestrates everything
    dot.edge("runner", "tasks", label="generates\nqueries", color=C_BENCH, fontcolor="#155E75")
    dot.edge("tasks", "langgraph", color=C_BENCH, style="dashed")
    dot.edge("tasks", "langchain", color=C_BENCH, style="dashed")
    dot.edge("tasks", "autogen", color=C_BENCH, style="dashed")
    dot.edge("runner", "inj_main", label="configures\nfailures", color=C_INJECTOR, fontcolor="#92400E")
    dot.edge("classifier", "metrics", label="classifications", color=C_OBSERVER, fontcolor="#4C1D95")
    dot.edge("runner", "metrics", label="RunResults", color=C_BENCH, fontcolor="#155E75")

    # Metrics → Outputs
    dot.edge("metrics", "csv", color=C_OUTPUT)
    dot.edge("metrics", "plots", color=C_OUTPUT)
    dot.edge("plots", "paper", color=C_OUTPUT, style="dashed")

    _save(dot, "architecture_diagram")


# ═══════════════════════════════════════════════════════════════════════
# 2. EXPERIMENT FLOWCHART
# ═══════════════════════════════════════════════════════════════════════

def experiment_flowchart() -> None:
    dot = graphviz.Digraph("flowchart", engine="dot")
    dot.attr(
        rankdir="TB", bgcolor=C_BG, pad="0.5", nodesep="0.5", ranksep="0.55",
        label="AgentProbe — Experiment Pipeline",
        labelloc="t", fontsize="22", fontname=TITLE_FONT, fontcolor="#111827",
    )
    dot.attr("node", fontname=FONT, fontsize="11", margin="0.2,0.12", penwidth="1.5")
    dot.attr("edge", fontname=FONT, fontsize="9", color="#6B7280", penwidth="1.3")

    # Start / End
    dot.node("start", "START", shape="circle", style="filled",
             fillcolor="#10B981", fontcolor=C_WHITE, width="0.8")
    dot.node("end", "END", shape="doublecircle", style="filled",
             fillcolor="#EF4444", fontcolor=C_WHITE, width="0.8")

    # Decision diamonds
    diamond = dict(shape="diamond", style="filled", fillcolor="#FEF3C7",
                   fontcolor="#78350F", margin="0.15,0.1")

    # Process boxes
    def proc(name, label, color, fcolor=C_WHITE):
        dot.node(name, label, shape="box", style="rounded,filled",
                 fillcolor=color, fontcolor=fcolor)

    def io_box(name, label):
        dot.node(name, label, shape="parallelogram", style="filled",
                 fillcolor="#E5E7EB", fontcolor="#1F2937")

    # ── Nodes ──
    io_box("config", "Load CLI Config\n--frameworks --tasks\n--failure-types --n-runs")
    proc("build_llm", "Build LLM\nOllama / OpenAI / Anthropic", C_LLM)

    dot.node("loop_fw", "Next\nFramework?", **diamond)
    proc("build_agent", "Build Agent\nLangGraph | LangChain | AutoGen", C_AGENT_LG)

    dot.node("loop_task", "Next\nTask?", **diamond)
    proc("gen_task", "Generate Task Queries\ntool_use / rag / multi_agent\n(10 queries each)", "#0891B2")

    dot.node("loop_fail", "Next Failure\nType?", **diamond)
    proc("configure_inj", "Configure Injector\nnone / timeout / malformed\nrate_limit / api_error", C_INJECTOR)

    proc("run_n", "Run N=10 Trials\nBenchmarkRunner.run()", "#2563EB")

    # Detail subgraph for single trial
    with dot.subgraph(name="cluster_trial") as s:
        s.attr(label="  Single Trial  ", style="rounded,dashed", color="#9CA3AF",
               fontname=TITLE_FONT, fontsize="11", fontcolor="#6B7280",
               bgcolor="#F9FAFB")
        s.node("invoke", "Invoke Agent\nwith query", shape="box", style="rounded,filled",
               fillcolor="#DBEAFE", fontcolor="#1E40AF", fontname=FONT, fontsize="10")
        s.node("inject", "Failure Injected?\n(probabilistic)", shape="diamond", style="filled",
               fillcolor="#FEF3C7", fontcolor="#78350F", fontname=FONT, fontsize="9",
               margin="0.1,0.08")
        s.node("record_ok", "Record\nRunResult(success)", shape="box", style="rounded,filled",
               fillcolor="#D1FAE5", fontcolor="#065F46", fontname=FONT, fontsize="9")
        s.node("classify", "Classify Failure\ncategory + severity", shape="box", style="rounded,filled",
               fillcolor="#DDD6FE", fontcolor="#4C1D95", fontname=FONT, fontsize="9")
        s.node("retry", "Attempt Recovery\n(1 retry)", shape="box", style="rounded,filled",
               fillcolor="#FEE2E2", fontcolor="#991B1B", fontname=FONT, fontsize="9")
        s.node("record_fail", "Record RunResult\n(failure + recovery + MTTR)", shape="box", style="rounded,filled",
               fillcolor="#FEF3C7", fontcolor="#78350F", fontname=FONT, fontsize="9")

    proc("compute", "Compute Metrics\nfailure_rate · recovery_rate\nMTTR · P50/P90/P99", "#7C3AED")
    io_box("save", "Save Results\nCSV + JSON + Plots")

    # ── Edges ──
    dot.edge("start", "config")
    dot.edge("config", "build_llm")
    dot.edge("build_llm", "loop_fw")

    dot.edge("loop_fw", "build_agent", label="yes", fontcolor="#10B981", color="#10B981")
    dot.edge("build_agent", "loop_task")
    dot.edge("loop_task", "gen_task", label="yes", fontcolor="#10B981", color="#10B981")
    dot.edge("gen_task", "loop_fail")
    dot.edge("loop_fail", "configure_inj", label="yes", fontcolor="#10B981", color="#10B981")
    dot.edge("configure_inj", "run_n")

    # Into trial detail
    dot.edge("run_n", "invoke", label="per trial", style="dashed")
    dot.edge("invoke", "inject")
    dot.edge("inject", "record_ok", label="no", fontcolor="#10B981", color="#10B981")
    dot.edge("inject", "classify", label="yes", fontcolor="#EF4444", color="#EF4444")
    dot.edge("classify", "retry")
    dot.edge("retry", "record_fail")

    # Back from trial
    dot.edge("record_ok", "compute", style="dashed")
    dot.edge("record_fail", "compute", style="dashed")
    dot.edge("compute", "save")

    # Loop backs
    dot.edge("save", "loop_fail", label="next failure type", style="dotted", color="#9CA3AF", constraint="false")
    dot.edge("loop_fail", "loop_task", label="no more\nfailures", fontcolor="#EF4444",
             color="#EF4444", style="dashed", constraint="false")
    dot.edge("loop_task", "loop_fw", label="no more\ntasks", fontcolor="#EF4444",
             color="#EF4444", style="dashed", constraint="false")
    dot.edge("loop_fw", "end", label="done", fontcolor="#EF4444", color="#EF4444")

    # Annotation
    dot.node("note", "Experiment Matrix\n─────────────────\n3 Frameworks\n× 3 Tasks\n× 5 Failure Types\n× 10 Runs\n= 450 total trials",
             shape="note", style="filled", fillcolor="#FEF9C3", fontcolor="#713F12",
             fontname=FONT, fontsize="10")
    dot.edge("note", "run_n", style="invis")

    _save(dot, "experiment_flowchart")


# ═══════════════════════════════════════════════════════════════════════
# 3. FAILURE TAXONOMY
# ═══════════════════════════════════════════════════════════════════════

def failure_taxonomy() -> None:
    dot = graphviz.Digraph("taxonomy", engine="dot")
    dot.attr(
        rankdir="LR", bgcolor=C_BG, pad="0.5", nodesep="0.3", ranksep="1.0",
        label="AgentProbe — Failure Taxonomy (5 Categories)",
        labelloc="t", fontsize="22", fontname=TITLE_FONT, fontcolor="#111827",
    )
    dot.attr("node", fontname=FONT, fontsize="11", penwidth="1.5")
    dot.attr("edge", color="#9CA3AF", penwidth="1.3", arrowsize="0.7")

    # Root
    dot.node("root", "Agent Failure\nTaxonomy", shape="box", style="rounded,filled,bold",
             fillcolor="#1F2937", fontcolor=C_WHITE, fontsize="14", fontname=TITLE_FONT,
             margin="0.3,0.2")

    categories = [
        ("tool", "TOOL_CALL", C_INJECTOR, "#78350F",
         [("timeout", "Timeout\n(configurable delay)"),
          ("malformed", "Malformed Output\n(corrupted response)"),
          ("rate_limit", "Rate Limit\n(429 + Retry-After)"),
          ("api_error", "API Error\n(5xx status codes)")]),

        ("orch", "ORCHESTRATION", "#EF4444", C_WHITE,
         [("inf_loop", "Infinite Loop\n(stuck in cycle)"),
          ("wrong_branch", "Wrong Branch\n(incorrect routing)"),
          ("state_corrupt", "State Corruption\n(mangled state)"),
          ("missing_handoff", "Missing Handoff\n(lost delegation)")]),

        ("ctx", "CONTEXT", "#2563EB", C_WHITE,
         [("overflow", "Context Overflow\n(exceeds window)"),
          ("lost_state", "Lost State\n(amnesia mid-task)"),
          ("halluc_hist", "Hallucinated History\n(fabricated context)")]),

        ("lat", "LATENCY", "#7C3AED", C_WHITE,
         [("p99_spike", "P99 Spike\n(tail latency)"),
          ("cascade", "Cascading Slow\n(propagating delay)"),
          ("cold_start", "Cold Start\n(initialization lag)"),
          ("jitter", "Jitter\n(variable delay)")]),

        ("consist", "CONSISTENCY", "#0891B2", C_WHITE,
         [("nondet", "Non-Deterministic\n(varying outputs)"),
          ("drift", "Semantic Drift\n(meaning shift)")]),
    ]

    for cat_id, cat_label, cat_color, cat_fcolor, subtypes in categories:
        dot.node(cat_id, cat_label, shape="box", style="rounded,filled,bold",
                 fillcolor=cat_color, fontcolor=cat_fcolor, fontsize="12",
                 fontname=TITLE_FONT, margin="0.25,0.15")
        dot.edge("root", cat_id, penwidth="2", color=cat_color)

        for sub_id, sub_label in subtypes:
            full_id = f"{cat_id}_{sub_id}"
            # Lighter version of category color for subtypes
            dot.node(full_id, sub_label, shape="box", style="rounded,filled",
                     fillcolor=C_LIGHT, fontcolor="#374151", fontsize="9",
                     margin="0.15,0.1")
            dot.edge(cat_id, full_id, color=cat_color, arrowsize="0.5")

    # Legend
    dot.node("legend", (
        "Legend\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "Injector modules: Tool, Orchestration,\n"
        "Context, Latency\n"
        "Classifier-only: Consistency\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "Each subtype is configurable via\n"
        "ToolFailureConfig / *FailureConfig"
    ), shape="note", style="filled", fillcolor="#FEF9C3", fontcolor="#713F12",
       fontsize="9", fontname=FONT)

    _save(dot, "failure_taxonomy")


# ═══════════════════════════════════════════════════════════════════════
# 4. FRAMEWORK INTERNALS COMPARISON
# ═══════════════════════════════════════════════════════════════════════

def framework_comparison() -> None:
    dot = graphviz.Digraph("frameworks", engine="dot")
    dot.attr(
        rankdir="TB", bgcolor=C_BG, pad="0.5", nodesep="0.4", ranksep="0.6",
        label="Agent Framework Internals — Side-by-Side Comparison",
        labelloc="t", fontsize="22", fontname=TITLE_FONT, fontcolor="#111827",
    )
    dot.attr("node", fontname=FONT, fontsize="10", penwidth="1.5",
             shape="box", style="rounded,filled", margin="0.2,0.1")
    dot.attr("edge", fontname=FONT, fontsize="8", penwidth="1.2")

    # ── LangGraph ──
    with dot.subgraph(name="cluster_lg") as s:
        s.attr(label="LangGraph (ReAct + StateGraph)", style="rounded,filled",
               color="#16A34A", fillcolor="#F0FDF4",
               fontname=TITLE_FONT, fontsize="13", fontcolor="#14532D")

        s.node("lg_input", "User Query", fillcolor="#BBF7D0", fontcolor="#14532D")
        s.node("lg_state", "AgentState\n(messages list)", fillcolor=C_AGENT_LG, fontcolor=C_WHITE)
        s.node("lg_llm", "LLM Node\n(bind_tools)", fillcolor="#22C55E", fontcolor=C_WHITE)
        s.node("lg_decide", "Should\nContinue?", shape="diamond", fillcolor="#FEF3C7", fontcolor="#78350F")
        s.node("lg_tools", "Tool Node\n(web_search, calc)", fillcolor="#86EFAC", fontcolor="#14532D")
        s.node("lg_inject", "Injector\n(wraps tools)", fillcolor=C_INJECTOR, fontcolor=C_WHITE, fontsize="9")
        s.node("lg_output", "Final Response", fillcolor="#BBF7D0", fontcolor="#14532D")

        s.edge("lg_input", "lg_state")
        s.edge("lg_state", "lg_llm", label="invoke")
        s.edge("lg_llm", "lg_decide")
        s.edge("lg_decide", "lg_tools", label="tool_call", color="#10B981")
        s.edge("lg_decide", "lg_output", label="end", color="#EF4444")
        s.edge("lg_tools", "lg_inject", style="dashed", color=C_INJECTOR)
        s.edge("lg_tools", "lg_state", label="update\nstate", constraint="false", color="#6B7280")

    # ── LangChain ──
    with dot.subgraph(name="cluster_lc") as s:
        s.attr(label="LangChain (AgentExecutor)", style="rounded,filled",
               color="#2563EB", fillcolor="#EFF6FF",
               fontname=TITLE_FONT, fontsize="13", fontcolor="#1E3A5F")

        s.node("lc_input", "User Query", fillcolor="#BFDBFE", fontcolor="#1E3A5F")
        s.node("lc_prompt", "ChatPromptTemplate\n+ agent_scratchpad", fillcolor=C_AGENT_LC, fontcolor=C_WHITE)
        s.node("lc_llm", "LLM\n(tool_calling_agent)", fillcolor="#3B82F6", fontcolor=C_WHITE)
        s.node("lc_parse", "Parse\nAction", shape="diamond", fillcolor="#FEF3C7", fontcolor="#78350F")
        s.node("lc_tools", "@tool decorated\n(web_search, calc)", fillcolor="#93C5FD", fontcolor="#1E3A5F")
        s.node("lc_inject", "Injector\n(inject_langchain_tool)", fillcolor=C_INJECTOR, fontcolor=C_WHITE, fontsize="9")
        s.node("lc_output", "Final Answer", fillcolor="#BFDBFE", fontcolor="#1E3A5F")

        s.edge("lc_input", "lc_prompt")
        s.edge("lc_prompt", "lc_llm")
        s.edge("lc_llm", "lc_parse")
        s.edge("lc_parse", "lc_tools", label="action", color="#2563EB")
        s.edge("lc_parse", "lc_output", label="final_answer", color="#EF4444")
        s.edge("lc_tools", "lc_inject", style="dashed", color=C_INJECTOR)
        s.edge("lc_tools", "lc_prompt", label="observation\n→ scratchpad", constraint="false", color="#6B7280")

    # ── AutoGen ──
    with dot.subgraph(name="cluster_ag") as s:
        s.attr(label="AutoGen (Multi-Agent Chat)", style="rounded,filled",
               color="#DC2626", fillcolor="#FEF2F2",
               fontname=TITLE_FONT, fontsize="13", fontcolor="#7F1D1D")

        s.node("ag_input", "User Query", fillcolor="#FECACA", fontcolor="#7F1D1D")
        s.node("ag_user", "UserProxyAgent\nhuman_input=NEVER\ncode_exec=False", fillcolor=C_AGENT_AG, fontcolor=C_WHITE)
        s.node("ag_assist", "AssistantAgent\n(LLM-powered)", fillcolor="#F87171", fontcolor=C_WHITE)
        s.node("ag_func", "function_map\n(web_search, calc)", fillcolor="#FCA5A5", fontcolor="#7F1D1D")
        s.node("ag_inject", "Injector\n(wraps functions)", fillcolor=C_INJECTOR, fontcolor=C_WHITE, fontsize="9")
        s.node("ag_chat", "Multi-turn Chat\n(max_turns=2)", shape="diamond", fillcolor="#FEF3C7", fontcolor="#78350F")
        s.node("ag_output", "Last Assistant\nMessage", fillcolor="#FECACA", fontcolor="#7F1D1D")

        s.edge("ag_input", "ag_user")
        s.edge("ag_user", "ag_assist", label="initiate_chat", color="#DC2626")
        s.edge("ag_assist", "ag_func", label="function_call", color="#DC2626")
        s.edge("ag_func", "ag_inject", style="dashed", color=C_INJECTOR)
        s.edge("ag_func", "ag_user", label="result", constraint="false", color="#6B7280")
        s.edge("ag_assist", "ag_chat")
        s.edge("ag_chat", "ag_user", label="continue", color="#DC2626", constraint="false")
        s.edge("ag_chat", "ag_output", label="terminate", color="#EF4444")

    # ── Key difference annotations ──
    dot.node("note1", (
        "Key Differences\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "LangGraph: Explicit state graph, full control\n"
        "           over routing & tool execution\n\n"
        "LangChain: AgentExecutor loop, automatic\n"
        "           scratchpad management\n\n"
        "AutoGen:   Two-agent conversation, function\n"
        "           calls bypass tool-level injection\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "All agents share: same tools, same LLM,\n"
        "same failure configs, same benchmark tasks"
    ), shape="note", style="filled", fillcolor="#FEF9C3", fontcolor="#713F12",
       fontsize="10", fontname=FONT)

    _save(dot, "framework_comparison")


# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Generating diagrams...")
    architecture_diagram()
    experiment_flowchart()
    failure_taxonomy()
    framework_comparison()
    print("Done! All diagrams saved to experiments/results/plots/")


if __name__ == "__main__":
    main()
