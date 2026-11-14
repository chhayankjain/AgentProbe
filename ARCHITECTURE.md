# Architecture

This document explains AgentProbe's internal design to help contributors navigate the codebase.

## High-Level Flow

```
User Query → Agent Adapter → LLM → Tool Call → Fault Injector → Tool Execution → Response
                                        ↓                ↓
                                   OTel Tracer      Injection Log
                                        ↓                ↓
                                  Span Export      Metrics Calculator
```

The benchmark runner orchestrates this flow across multiple runs, frameworks, and fault conditions.

## Core Modules

### `agentprobe/agents/` — Framework Adapters

Each adapter wraps a specific agent framework behind a common `BaseAgent` interface:

```
BaseAgent (base.py)
├── LangGraphToolAgent (langgraph_agent.py)  — LangGraph ReAct agent
├── LangChainToolAgent (langchain_agent.py)  — LangChain AgentExecutor
└── AutoGenToolAgent (autogen_agent.py)      — AutoGen two-agent pipeline
```

**Key interface** (the `BaseAgent` protocol in `agents/base.py`):

```python
framework: str
invoke(input: dict) -> dict      # input["input"] is the query; returns {"output", "latency_ms", "framework", ...}
reset() -> None                  # clear conversation state (never the injector)
get_injector_summary() -> dict   # condition-wide injection stats
```

All adapters accept the same tools (calculator, web_search) and the same LLM configuration. The adapter is responsible for translating between the framework's native API and AgentProbe's common interface.

Note that `LangChainToolAgent` is built on `langchain.agents.create_agent`, which compiles to a LangGraph graph internally. The two adapters therefore share an execution path, which is why the paper reports identical LangGraph/LangChain failure patterns.

**Adding a new framework:** Write a class that satisfies `BaseAgent` (it is a structural `Protocol`, no subclassing needed), wrap your tools with `ToolFailureInjector`, and register the class in `build_agent()` in `experiments/run_experiment_v2.py`.

### `agentprobe/injector/` — Fault Injectors

Injectors intercept tool calls and probabilistically introduce failures:

```
ToolFailureInjector (tool_failure.py)        — timeout, malformed output, rate limit, API error
OrchestrationFailureInjector (orchestration_failure.py) — infinite loops, missing handoffs
ContextFailureInjector (context_failure.py)  — context overflow, history corruption
LatencyFailureInjector (latency_failure.py)  — P99 spikes, cascading delays
```

**How injection works:**
1. The injector wraps a tool function via `inject(fn)` → returns a wrapped function
2. On each call, a seeded RNG decides whether to inject (probability `p`)
3. If injecting: raises an exception or corrupts the response
4. If not: passes through to the real tool
5. Every decision is logged with `(call_index, injected, fault_type)`

**Key design choice:** Injectors are stateful (they track call count and RNG state) but the RNG is seeded for reproducibility.

### `agentprobe/benchmark/` — Experiment Infrastructure

```
runner.py      — BenchmarkRunner: orchestrates N runs, handles timeouts, retries, serialization
metrics.py     — MetricsCalculator: computes failure rate, recovery rate, MTTR, percentile latencies
statistics.py  — Wilson CIs, Cohen's d/h, comparison reports
tasks/         — Graded query corpora (25 tool_use, 15 rag, 15 multi_agent); NOT the paper's prompts
grader.py      — Answer-string presence checker (substring match for assessable queries)
metadata.py    — Captures experiment metadata (model, seed, git commit, pip freeze)
```

**Two prompt sets.** The runner's default `task_fn` is `paper_task_fn`, which cycles the five prompts per task in `runner.PAPER_PROMPTS`. Every batch reported in the paper used this set. The larger graded corpora in `benchmark/tasks/` are available via `corpus_task_fn(task)` but have not been used in a published experiment.

**Run lifecycle:**
1. The experiment script creates the agent, which creates its own injector for the condition
2. For each of N runs: invoke agent, catch exceptions, record success/failure/latency
3. On failure: attempt recovery (the harness re-invokes the agent once, injector still active)
4. Serialize results to JSON + CSV
5. Compute summary metrics

**Metric definitions.** "Failure" means the agent invocation raised an exception that reached the harness. "Recovery" means the harness's single re-invocation succeeded; it is a harness-level retry, not agent-internal self-repair. MTTR is the mean latency of successful re-invocations. P99 is ceiling nearest-rank, so for n < 100 it equals the sample maximum.

### `agentprobe/observer/` — Observability

```
tracer.py      — OpenTelemetry span creation and export
classifier.py  — Rule-based failure classification (maps exceptions → taxonomy categories)
dashboard.py   — Prometheus metrics exporter (counters, gauges, histograms)
```

The tracer instruments each agent invocation with OTel spans. The classifier post-processes spans to assign failure categories from the taxonomy.

## Experiment Scripts

```
experiments/
├── run_experiment_v2.py        — Main experiment runner (Phase 2 protocol)
├── run_qualification.py        — Deterministic validation (p=1.0, known-answer tasks)
├── run_benchmarks.py           — Phase 1 runner (legacy, broader task matrix)
├── smoke_test_model.py         — Quick tool-calling sanity check
├── autogen_diagnostic.py       — Direct client vs AutoGen shim comparison
├── analyze_results.py          — Statistical analysis for paper tables
├── analyze_cross_model.py      — Cross-model comparison analysis
├── generate_paper_figures.py   — Data-driven figure generation
└── archive_v1/                 — Archived Phase 1 scripts and one-time utilities
```

## Data Flow

```
run_experiment_v2.py
  → results_v2/{timestamp}/
      ├── {framework}_{task}_{fault_type}.json   — per-condition trial records
      ├── {framework}_{task}_{fault_type}.csv    — same data, tabular
      ├── combined_summary.json                   — aggregated metrics
      ├── experiment_metadata.json                — reproducibility metadata
      ├── checkpoint.json                         — resume state
      └── manifest.json                           — file inventory

analyze_results.py / analyze_cross_model.py
  → reads combined_summary.json files
  → produces paper-ready tables and statistics

generate_paper_figures.py
  → reads batch data
  → outputs PDF figures to experiments/plots/
```

## Key Design Decisions

1. **Adapters, not plugins.** Each framework adapter is a concrete class, not a plugin system. This keeps the code simple and debuggable at the cost of requiring a code change to add a framework.

2. **Injection at the tool boundary.** Faults are injected by wrapping tool functions, not by modifying LLM responses or framework internals. This ensures injection works identically across frameworks.

3. **Seeded RNG per injector.** Each injector instance has its own `random.Random(seed)`, making injection sequences reproducible regardless of framework-internal randomness.

4. **Checkpoint-based resume.** Long experiments can be interrupted and resumed. The checkpoint stores structured triples `(framework, task, fault_type)` with validation on reload.

5. **Metrics are pure computation.** `MetricsCalculator` takes a list of run results and returns metrics. No side effects, no state, easy to test.

## Testing

```
tests/
├── test_benchmark_runner.py       — runner orchestration, timeout handling
├── test_metrics.py                — metrics computation edge cases
├── test_experiment_state.py       — checkpoint validation and resume safety
├── test_pilot_hung.py             — hung invocation detection
├── test_qualification_checker.py  — answer validation logic
├── test_tool_failure.py           — injection probability and fault types
├── test_public_api.py             — public exports, BaseAgent protocol, config/injector compatibility
├── test_metadata.py               — metadata capture
└── ...
```

Tests use mocked agents and tools — no LLM calls required. Run with `pytest tests/ -v`.
