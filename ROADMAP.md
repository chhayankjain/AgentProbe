# AgentProbe — Build Roadmap

This document tracks the incremental build plan for AgentProbe. Each chunk is a
self-contained deliverable with its own tests. Chunks are built in strict dependency
order so every slice is immediately runnable and verifiable.

---

## Progress

| Chunk | Name | Status | Tests |
|---|---|---|---|
| 1 | Config + Core Schemas | ✅ Complete | 29 passing |
| 2 | Failure Injectors | 🔲 Next | — |
| 3 | Agent Adapters | 🔲 Planned | — |
| 4 | Benchmark Suite | 🔲 Planned | — |
| 5 | Observer Extras | 🔲 Planned | — |
| 6 | CLI + Experiment Entry Point | 🔲 Planned | — |

**Foundation (pre-roadmap):** `agentprobe/observer/tracer.py` — OpenTelemetry tracing (6 tests ✅)

---

## Chunk 1 — Config + Core Schemas ✅

**Goal:** Establish the type contracts that every downstream module imports. No agent
adapter or injector can be written without these. Zero external dependencies — purely
Pydantic models and stdlib dataclasses.

### Files delivered

| File | Purpose |
|---|---|
| `agentprobe/config.py` | Frozen `Settings` dataclass loaded from env variables. Module-level `settings` singleton. `require_openai()` / `require_anthropic()` raise `RuntimeError` with an actionable message when keys are absent. |
| `agentprobe/injector/types.py` | `ToolFailureType` enum (NONE, TIMEOUT, MALFORMED\_OUTPUT, RATE\_LIMIT, API\_ERROR, SERVER\_ERROR) and `ToolFailureConfig` Pydantic model. |
| `agentprobe/benchmark/types.py` | `TaskType` enum, `AgentFramework` enum, and Pydantic models: `RunResult`, `BenchmarkConfig`, `BenchmarkResults`, `BenchmarkMetrics`. |
| `agentprobe/agents/base.py` | `AgentResult` Pydantic model and `BaseAgent` runtime-checkable Protocol. |
| `tests/test_schemas.py` | 29 tests — validation constraints, serialization round-trips, Protocol checking, all `Settings` env-var paths. |

### How to verify

```bash
pytest tests/test_schemas.py -v       # 29 tests
pytest tests/ -v -m "not slow and not integration"  # 35 total (includes tracer)
```

---

## Chunk 2 — Failure Injectors 🔲

**Goal:** Pure-Python wrappers that intercept tool calls and simulate the five failure
categories. No LLM required — fully testable with mocked callables.

### Files planned

| File | Purpose |
|---|---|
| `agentprobe/injector/tool_failure.py` | `ToolFailureInjector` — intercepts calls based on `ToolFailureConfig`; raises simulated errors (timeout, 429, 400, 500) or returns malformed output. Uses seeded `random.Random` for reproducibility. |
| `agentprobe/injector/latency_failure.py` | `LatencyInjector` — adds Gaussian jitter and optional P99 spikes before delegating to the real tool. |
| `agentprobe/injector/orchestration_failure.py` | `OrchestrationFailureInjector` — simulates infinite loops (step counter) and dropped handoffs (empty output). |
| `agentprobe/injector/context_failure.py` | `ContextFailureInjector` — truncates or corrupts the message history passed to an agent. |
| `tests/test_injectors.py` | Unit tests with mocked callables — no HTTP, no LLM. |

**Key design:** all injectors wrap a `Callable` and are composable. The `seed` field on
`ToolFailureConfig` ensures identical failure sequences across runs.

---

## Chunk 3 — Agent Adapters 🔲

**Depends on:** Chunks 1 + 2

**Goal:** Concrete implementations of `BaseAgent` for LangGraph, LangChain, and AutoGen.
The shared protocol lets the benchmark runner swap frameworks without changing task code.

### Files planned

| File | Purpose |
|---|---|
| `agentprobe/agents/langgraph_agent.py` | `LangGraphToolAgent` — ReAct graph that accepts `BaseTool` instances (optionally wrapped by `ToolFailureInjector`). Implements `run()` and `reset()`. |
| `agentprobe/agents/langchain_agent.py` | `LangChainAgentExecutor` — wraps LangChain `AgentExecutor`. |
| `agentprobe/agents/autogen_agent.py` | `AutoGenAgent` — wraps AutoGen two-agent pipeline (UserProxy + Assistant). |
| `tests/test_agents.py` | Tests with mocked LLMs (no real API calls). |

---

## Chunk 4 — Benchmark Suite 🔲

**Depends on:** Chunks 1–3

**Goal:** Task definitions, a multi-run experiment runner, and a metrics calculator.
`MetricsCalculator` is a pure function testable without any LLM.

### Files planned

| File | Purpose |
|---|---|
| `agentprobe/benchmark/tasks/tool_use.py` | Tool-use task spec (e.g. weather lookup + arithmetic). |
| `agentprobe/benchmark/tasks/rag.py` | RAG task spec (retrieve + synthesize). |
| `agentprobe/benchmark/tasks/multi_agent.py` | Multi-agent coordination task spec. |
| `agentprobe/benchmark/runner.py` | `BenchmarkRunner` — runs `n_runs`, records `RunResult` per run, traces with OTel, optionally saves JSON. |
| `agentprobe/benchmark/metrics.py` | `MetricsCalculator` — computes failure rate, MTTR, P99/P50 latency from `BenchmarkResults`. |
| `tests/test_benchmark.py` | Pure-computation tests for `MetricsCalculator`; smoke test for `BenchmarkRunner` with a stub agent. |

---

## Chunk 5 — Observer Extras 🔲

**Depends on:** Chunks 1–4

**Goal:** Complete the observability stack with a rule-based failure classifier and a
Prometheus metrics exporter.

### Files planned

| File | Purpose |
|---|---|
| `agentprobe/observer/classifier.py` | `FailureClassifier` — maps `RunResult` error fields to the 5-category taxonomy; returns a `FailureReport`. |
| `agentprobe/observer/dashboard.py` | `PrometheusExporter` — starts a `/metrics` HTTP server; exposes `agentprobe_failure_rate`, `agentprobe_mttr_ms`, `agentprobe_run_duration_ms` (Histogram). |

---

## Chunk 6 — CLI + Experiment Entry Point 🔲

**Depends on:** Chunks 1–5

**Goal:** A user-facing `agentprobe` CLI command and a standalone experiment script
matching the README quick-start invocation.

### Files planned

| File | Purpose |
|---|---|
| `agentprobe/cli.py` | Typer app with `run` command. Parses args → `BenchmarkConfig` → `BenchmarkRunner.run()` → prints `BenchmarkMetrics` via `rich`. |
| `experiments/run_benchmarks.py` | Standalone script for notebook/scripting use. Thin wrapper over CLI logic. |

### End-to-end smoke test (requires Ollama)

```bash
python experiments/run_benchmarks.py \
    --frameworks langgraph --tasks tool_use \
    --failure-types timeout --n-runs 5
```

---

## Architecture at a Glance

```
agentprobe/
├── config.py                        ✅  Centralized env-based settings
├── observer/
│   ├── tracer.py                    ✅  OpenTelemetry tracing
│   ├── classifier.py                🔲  Rule-based failure classification
│   └── dashboard.py                 🔲  Prometheus metrics exporter
├── injector/
│   ├── types.py                     ✅  ToolFailureType + ToolFailureConfig
│   ├── tool_failure.py              🔲  Tool call failure injection
│   ├── latency_failure.py           🔲  Latency / jitter injection
│   ├── orchestration_failure.py     🔲  Loop / handoff failure injection
│   └── context_failure.py           🔲  Context window failure injection
├── agents/
│   ├── base.py                      ✅  BaseAgent Protocol + AgentResult
│   ├── langgraph_agent.py           🔲  LangGraph ReAct adapter
│   ├── langchain_agent.py           🔲  LangChain AgentExecutor adapter
│   └── autogen_agent.py             🔲  AutoGen two-agent adapter
└── benchmark/
    ├── types.py                     ✅  BenchmarkConfig / RunResult / Metrics
    ├── runner.py                    🔲  Multi-run experiment orchestration
    ├── metrics.py                   🔲  MTTR / failure rate / P99 computation
    └── tasks/
        ├── tool_use.py              🔲  Tool-use task definition
        ├── rag.py                   🔲  RAG task definition
        └── multi_agent.py           🔲  Multi-agent task definition
```

---

## Coding Conventions

All new files must follow the patterns established in `observer/tracer.py`:

| Rule | Detail |
|---|---|
| `from __future__ import annotations` | First line in every module |
| Union types | `str \| None`, not `Optional[str]` |
| Docstrings | Module docstring with `::` usage example; Google-style Args/Returns on classes |
| `__all__` | Declared in every `__init__.py` |
| Line length | 100 characters (ruff enforced) |
| Pydantic | v2 API (`Field(default=...)`, `model_dump()`, `model_validate()`) |
| Tests | `class Test<Subject>:` with `test_` methods |
| Mypy | Strict mode — no untyped functions |
