# AgentProbe

[![CI](https://github.com/chhayankjain/AgentProbe/actions/workflows/ci.yml/badge.svg)](https://github.com/chhayankjain/AgentProbe/actions/workflows/ci.yml)
[![CodeQL](https://github.com/chhayankjain/AgentProbe/actions/workflows/codeql.yml/badge.svg)](https://github.com/chhayankjain/AgentProbe/actions/workflows/codeql.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/agentprobe-toolkit.svg)](https://pypi.org/project/agentprobe-toolkit/)
[![GitHub release](https://img.shields.io/github/v/release/chhayankjain/AgentProbe)](https://github.com/chhayankjain/AgentProbe/releases)

**Fault injection and observability toolkit for LLM agents**

AgentProbe wraps an agent's tools with seeded, logged fault injectors (timeouts, malformed output, rate limits, API errors), traces every invocation and tool call with OpenTelemetry, and reports failure rate, recovery rate, MTTR, and latency percentiles across frameworks. It ships adapters for LangGraph, LangChain, and AutoGen, plus the scripts and data behind the companion paper.

Companion to the paper: *"AgentProbe: Systematic Fault Injection and Observability for Evaluating the Reliability of LLM-Based Intelligent Agents"* (submitted to MAKE, MDPI, 2026).

---

## Failure Taxonomy

AgentProbe organises agent failures into five categories. Only the first has been exercised in a published experiment; the status column says exactly what exists for each.

| # | Category | Examples | Status in this repo |
|---|----------|----------|---------------------|
| 1 | **Tool Call** | API timeouts, malformed output, rate limits (HTTP 429), server errors (5xx) | Injector implemented, **used in all paper experiments** (`injector/tool_failure.py`) |
| 2 | **Orchestration** | Infinite loops, wrong branching, state corruption, dropped handoffs | Injector implemented and unit-tested, not yet used in an experiment (`injector/orchestration_failure.py`) |
| 3 | **Context** | Context window overflow, lost history, hallucinated history | Injector implemented and unit-tested, not yet used in an experiment (`injector/context_failure.py`) |
| 4 | **Latency** | P99 spikes, cascading slowdowns, jitter | Injector implemented; latency in the paper is a by-product of tool faults, not independently injected (`injector/latency_failure.py`) |
| 5 | **Consistency** | Non-deterministic outputs, divergent multi-agent state | Detected only, via a Jaccard-similarity classifier; nothing is injected (`observer/classifier.py`) |

Note that the current "malformed output" injector raises an exception at the tool boundary rather than returning corrupted data to the model, so it tests the agent's exception path, not its parsing. See issue [#28](https://github.com/chhayankjain/AgentProbe/issues/28).

---

## Installation

```bash
# From PyPI (LangGraph + LangChain adapters, local Ollama backend)
pip install "agentprobe-toolkit[ollama]"

# From a clone, for development or to run the paper's experiments
git clone https://github.com/chhayankjain/AgentProbe.git && cd AgentProbe
pip install -e ".[dev,plotting,ollama]"

# The AutoGen adapter targets pyautogen 0.1.14 and needs openai<1.0, which
# conflicts with langchain-openai. Install it in its own environment:
#   pip install "agentprobe-toolkit[ollama,autogen]"
```

The distribution is named `agentprobe-toolkit` because `agentprobe` was already taken on PyPI by an unrelated project. The import name is `agentprobe`.

## Quick Start

```bash

# Start Ollama with Llama 3.1 (used in all paper experiments)
ollama pull llama3.1

# For cross-model validation (optional)
# ollama pull mistral:7b
# ollama pull qwen2.5:7b

# Run baseline benchmark (no failures)
python experiments/run_benchmarks.py --frameworks langgraph --tasks tool_use --baseline-only

# Run with failure injection
python experiments/run_benchmarks.py \
    --frameworks langgraph langchain \
    --tasks tool_use rag \
    --failure-types none timeout rate_limit \
    --n-runs 50 \
    --save-results
```

## Programmatic Usage

```python
from langchain_ollama import ChatOllama

from agentprobe import (
    AgentProbeTracer,
    BenchmarkConfig,
    BenchmarkRunner,
    MetricsCalculator,
    ToolFailureConfig,
    ToolFailureType,
)
from agentprobe.agents.langgraph_agent import LangGraphToolAgent

# 1. Set up tracing (console by default; OTLP endpoint optional)
tracer = AgentProbeTracer(service_name="my-experiment", otlp_endpoint="http://localhost:4317")

# 2. Configure failure injection
failure_config = ToolFailureConfig(
    failure_type=ToolFailureType.TIMEOUT,
    failure_probability=0.2,    # 20% of tool calls will time out
    timeout_seconds=5.0,
    seed=42,                    # reproducible injection sequence
)

# 3. Build an agent; it wraps its tools with a ToolFailureInjector internally
llm = ChatOllama(model="llama3.1", temperature=0.0)
agent = LangGraphToolAgent(llm=llm, failure_config=failure_config)

# 4. Run the benchmark (defaults to the paper's five-prompt set)
config = BenchmarkConfig(
    framework="langgraph",
    task="tool_use",
    n_runs=50,
    failure_config=failure_config,
    recovery_attempts=1,
    save_results_path="experiments/results/langgraph_tool_use_timeout",
)
results = BenchmarkRunner(agent=agent, config=config).run()

# 5. Compute metrics
metrics = MetricsCalculator().compute(results)
print(f"Failure rate: {metrics.failure_rate:.1%}")
print(f"Recovery rate: {metrics.recovery_rate}")
print(f"MTTR: {metrics.mttr_ms:.0f}ms")
print(f"P99 latency: {metrics.p99_latency_ms:.0f}ms")
print(agent.get_injector_summary())
```

### Injecting faults into your own tools

The injector is framework-agnostic. Wrap any callable, or any LangChain `BaseTool`:

```python
from agentprobe import ToolFailureConfig, ToolFailureInjector, ToolFailureType

injector = ToolFailureInjector(
    ToolFailureConfig(failure_type=ToolFailureType.RATE_LIMIT, failure_probability=0.3, seed=7)
)

@injector.wrap
def lookup_price(sku: str) -> float:
    ...

faulty_tool = injector.inject_langchain_tool(my_langchain_tool)   # in-place, keeps schema
print(injector.summary())        # total_calls, injected_failures, failures_by_type, ...
print(injector.injection_log)    # one record per call with args, return/exception, span_id
```

### Two prompt sets

- **Paper prompts (default).** `BenchmarkRunner` cycles the five prompts per task in `agentprobe.benchmark.runner.PAPER_PROMPTS`. Every batch in the paper used these.
- **Graded corpora.** `agentprobe.benchmark.tasks` holds 25 tool-use, 15 RAG, and 15 multi-agent queries with keyword graders. They have not been used in a published experiment. Run them with `task_fn=corpus_task_fn("tool_use")` or `run_benchmarks.py --query-set corpus`.

### Metric definitions

- **Failure**: the agent invocation raised an exception that reached the harness. A wrong-but-exception-free answer is *not* a failure here; use the grader for that.
- **Recovery**: the harness re-invoked the agent once (injector still active) and that call succeeded. This is a harness-level retry, not agent-internal self-repair.
- **MTTR**: mean latency of successful re-invocations, in ms.
- **P50/P90/P99**: ceiling nearest-rank percentiles of per-run latency. For n < 100, P99 equals the sample maximum. Injected timeouts sleep for real, so latency under the timeout condition includes the injected delay.
- **Recovery rate** is `None` when a condition had no failures.

## Architecture

```
agentprobe/
├── injector/
│   ├── tool_failure.py          # Category 1: tool call failures
│   ├── orchestration_failure.py # Category 2: orchestration failures
│   ├── context_failure.py       # Category 3: context failures
│   └── latency_failure.py       # Category 4: latency failures
├── observer/
│   ├── tracer.py                # OpenTelemetry setup + span utilities
│   ├── classifier.py            # Rule-based failure classification
│   └── dashboard.py             # Prometheus metrics exporter
├── benchmark/
│   ├── runner.py                # BenchmarkRunner + PAPER_PROMPTS (the paper's prompt set)
│   ├── tasks/                   # Larger graded corpora (not used in the paper)
│   ├── metrics.py               # Failure rate, recovery rate, MTTR, percentiles
│   ├── grader.py                # Keyword / answer-presence grading
│   ├── statistics.py            # Wilson CIs, Cohen's d/h, comparisons
│   └── metadata.py              # Reproducibility metadata capture
└── agents/
    ├── base.py                  # BaseAgent protocol: invoke(dict) -> dict, reset()
    ├── langgraph_agent.py       # LangGraph ReAct agent
    ├── langchain_agent.py       # LangChain create_agent (compiles to LangGraph internally)
    └── autogen_agent.py         # AutoGen two-agent pipeline (pyautogen 0.1.14 + compat shim)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the data flow and design decisions.

## Observability Stack

```bash
# Start Prometheus + Grafana + Jaeger
docker compose up -d

# Grafana:    http://localhost:3000  (admin/admin)
# Jaeger:     http://localhost:16686
# Prometheus: http://localhost:9090
```

```python
from agentprobe.observer.dashboard import PrometheusExporter

exporter = PrometheusExporter(port=8000)
exporter.start()  # /metrics on :8000

# Auto-record from benchmark run
exporter.record_benchmark_run(results)
```

## Supported Frameworks & LLMs

| Framework | Adapter | Validated in paper | Notes |
|-----------|---------|--------------------|-------|
| LangGraph | `LangGraphToolAgent` | Yes (Llama 3.1, Mistral 7B, Qwen2.5 7B) | Hand-built ReAct graph |
| LangChain | `LangChainToolAgent` | Yes (same models) | Uses `create_agent`, which compiles to LangGraph; shares an execution path with the LangGraph adapter, so results converge |
| AutoGen | `AutoGenToolAgent` | Yes, with caveats | Targets **pyautogen 0.1.14** through a `functions`→`tools` compatibility shim. Phase 1 found it silently bypassed tools; Phase 2 patched that. Modern `ag2` / `autogen-agentchat` 0.4+ is not yet supported ([roadmap](https://github.com/chhayankjain/AgentProbe/issues/38)) |

| LLM Backend | Install extra | Validated in paper | Notes |
|-------------|---------------|--------------------|-------|
| Ollama (Llama 3.1 8B, Mistral 7B, Qwen2.5 7B, Q4_K_M) | `.[ollama]` | Yes | Local, zero API cost. The Phase 2, qualification, and diagnostic scripts are Ollama-only |
| OpenAI (GPT-4o etc.) | `.[openai]` | **No** | Wired into `run_benchmarks.py --llm openai` for LangGraph/LangChain only; never run for the paper. See [#26](https://github.com/chhayankjain/AgentProbe/issues/26) |
| Anthropic (Claude) | `.[anthropic]` | **No** | Same as OpenAI |

## Running Tests

```bash
pip install -e ".[dev,plotting]"   # from a clone
pytest tests/ -v

# Skip slow/integration tests
pytest tests/ -v -m "not slow and not integration"

# Lint
ruff check agentprobe/ tests/
```

## Reproducing Paper Results

The paper reports a two-phase study using Llama 3.1 8B (Q4_K_M) via Ollama, with injection seed 42 throughout.

### Phase 1 — Discovery (1,050 runs)

Phase 1 uses `run_benchmarks.py` with 30 runs for LangGraph/LangChain and 10 for AutoGen across 3 task types and 5 fault conditions. This phase discovered that AutoGen never invokes tools due to a legacy API-parameter mismatch.

> **Historical reproduction note:** The current codebase includes the patched AutoGen adapter (`_patch_openai_compat()`). To reproduce the original Phase 1 bypass behavior, use the pre-patch source at tag `v1.0.0`. Running Phase 1 commands with the current code will produce patched AutoGen behavior (tools are called), not the historical bypass.

```bash
# LangGraph / LangChain (30 runs per condition)
python experiments/run_benchmarks.py \
    --frameworks langgraph langchain \
    --tasks tool_use rag multi_agent \
    --failure-types none timeout malformed_output rate_limit api_error \
    --n-runs 30 --llm ollama --model llama3.1 --save-results

# AutoGen (10 runs per condition)
python experiments/run_benchmarks.py \
    --frameworks autogen \
    --tasks tool_use rag multi_agent \
    --failure-types none timeout malformed_output rate_limit api_error \
    --n-runs 10 --llm ollama --model llama3.1 --save-results
```

### Phase 2 — Patched Validation (450 runs)

Phase 2 uses `run_experiment_v2.py` with the patched AutoGen adapter (proper `tool_name=` recording, `_patch_openai_compat()` for the functions/tools protocol gap). Runs 30 trials per condition across 3 frameworks × 5 fault conditions on tool-use tasks only, with matched p=0.20 injection probability.

```bash
python experiments/run_experiment_v2.py \
    --frameworks langgraph langchain autogen \
    --model llama3.1 --seed 42
```

### Qualification Experiment (12 trials)

A small, targeted validation with deterministic calculator tasks and guaranteed fault injection (p=1.0). Produces a self-contained JSON artifact with inputs, outputs, tool events, injection log entries (with args/returns), and linked OTel trace span IDs.

```bash
python experiments/run_qualification.py \
    --frameworks langgraph langchain autogen \
    --model llama3.1
```

### Cross-Model Validation

Repeats the qualification and Phase 2 protocol on additional models (Mistral 7B, Qwen2.5 7B):

```bash
# Smoke test: verify tool calling works
python experiments/smoke_test_model.py --model mistral:7b
python experiments/smoke_test_model.py --model qwen2.5:7b

# Qualification (12 trials per model)
python experiments/run_qualification.py --model mistral:7b
python experiments/run_qualification.py --model qwen2.5:7b

# Full Phase 2 (450 runs per model)
python experiments/run_experiment_v2.py --model mistral:7b --seed 42
python experiments/run_experiment_v2.py --model qwen2.5:7b --seed 42

# Cross-model analysis
python experiments/analyze_cross_model.py \
    --batches experiments/results_v2/20261111_113514 experiments/results_v2/20261111_233435 experiments/results_v2/20261112_040156 \
    --models llama3.1 mistral:7b qwen2.5:7b
```

Results are saved to `experiments/results_v2/`.

### AutoGen Diagnostic (Direct Client vs Shim)

A diagnostic comparing the AutoGen adapter pipeline against a direct OpenAI-compatible client on the same three deterministic tasks. The compared paths differ in message format, stopping policy, and temperature handling; results are consistent with the adapter layer being a major contributor to low answer-string presence but do not isolate a single cause:

```bash
# Llama 3.1
python experiments/autogen_diagnostic.py --model llama3.1

# Qwen2.5 7B
python experiments/autogen_diagnostic.py --model qwen2.5:7b
```

Diagnostic artifacts are saved to `experiments/results_v2/diagnostic_*/`.

### Analysis

Phase 2 uses the corrected summaries under `experiments/results_v2/`. The original Phase 1 output (now at `experiments/archive_v1/results/combined_summary.json`) contains known counting bugs and is kept only for the historical record. The paper's tables are derived from:
- **Phase 2 batch (Llama 3.1):** `experiments/results_v2/20261111_113514/`
- **Phase 2 batch (Mistral 7B):** `experiments/results_v2/20261111_233435/`
- **Phase 2 batch (Qwen2.5 7B LG/LC):** `experiments/results_v2/20261112_040156/`
- **Phase 2 batch (Qwen2.5 7B AutoGen rerun):** `experiments/results_v2/20261112_063451/` (corrected extractor)
- **Qualification (Llama 3.1):** `experiments/results_v2/qualification_20261111_211913/` (12 trials, 10 passed)
- **Qualification (Mistral 7B):** `experiments/results_v2/qualification_20261111_233116/` (12 trials, 8 passed)
- **Qualification (Qwen2.5 7B):** `experiments/results_v2/qualification_20261111_233320/` (12 trials, 10 passed)

> **Source map:** The authoritative mapping from each paper table cell to its batch directory is in [`experiments/results_v2/paper_source_map.json`](experiments/results_v2/paper_source_map.json). The Qwen2.5 column combines two batches: LangGraph/LangChain from `20261112_040156` and AutoGen from `20261112_063451` (corrected output extractor). Llama and Mistral use their single original batches. The `analyze_cross_model.py` script does not automatically merge these; use the source map to select the correct batch per framework.

> **Model digests:** All three models use Q4_K_M quantization. Ollama digest prefixes: Llama `46e0c10c039e`, Mistral `6577803aa9a0`, Qwen `845dbda0ea48`. The `experiment_metadata.json` files in historical batches record `model_digest: unknown` because digest capture was added after the experiments ran; the digests listed here were identified post-hoc from the same Ollama installation.

## Citation

```bibtex
@misc{jain2026agentprobe,
  title={AgentProbe: Systematic Fault Injection and Observability for Evaluating the Reliability of LLM-Based Intelligent Agents},
  author={Jain, Chhayank and Pattnaik, Sarthak and Pinsky, Eugene},
  year={2026},
  note={Manuscript under review}
}
```

## License

MIT — see [LICENSE](LICENSE).
