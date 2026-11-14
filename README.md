# AgentProbe

**Reliability & Observability Toolkit for Production Agentic LLM Systems**

AgentProbe is an open source benchmarking toolkit that injects real-world failure modes into LangGraph, LangChain, and AutoGen agents, traces execution via OpenTelemetry, and measures reliability metrics (MTTR, failure rate, P99 latency) across frameworks and workloads.

Companion to the paper: *"AgentProbe: Systematic Fault Injection and Observability for Evaluating the Reliability of LLM-Based Intelligent Agents"* (submitted to MAKE, MDPI, 2026).

---

## Failure Taxonomy

AgentProbe classifies agentic failures into five categories:

| # | Category | Examples |
|---|----------|---------|
| 1 | **Tool Call** | API timeouts, malformed JSON output, rate limits (HTTP 429), server errors (5xx) |
| 2 | **Orchestration** | Infinite loops, wrong conditional branching, state corruption, dropped handoffs |
| 3 | **Context** | Context window overflow, lost message history, hallucinated history injection |
| 4 | **Latency** | P99 spikes, cascading slowdowns, cold start delays, network jitter |
| 5 | **Consistency** | Non-deterministic outputs, divergent multi-agent state |

---

## Quick Start

```bash
# Install
pip install -e ".[ollama]"

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
from agentprobe import ToolFailureInjector, AgentProbeTracer
from agentprobe.injector.tool_failure import ToolFailureConfig, ToolFailureType
from agentprobe.agents.langgraph_agent import LangGraphToolAgent
from agentprobe.benchmark.runner import BenchmarkRunner, BenchmarkConfig
from agentprobe.benchmark.metrics import MetricsCalculator

# 1. Set up tracing
tracer = AgentProbeTracer(
    service_name="my-experiment",
    otlp_endpoint="http://localhost:4317",  # optional
)

# 2. Configure failure injection
failure_config = ToolFailureConfig(
    failure_type=ToolFailureType.TIMEOUT,
    failure_probability=0.2,    # 20% of tool calls will timeout
    timeout_seconds=5.0,
    seed=42,                    # reproducible
)

# 3. Build agent
llm = ChatOllama(model="llama3.1", temperature=0.0)
agent = LangGraphToolAgent(llm=llm, failure_config=failure_config)

# 4. Run benchmark
config = BenchmarkConfig(
    framework="langgraph",
    task="tool_use",
    n_runs=50,
    failure_config=failure_config,
    recovery_attempts=1,
    save_results_path="experiments/results/langgraph_tool_use_timeout",
)
runner = BenchmarkRunner(agent=agent, config=config)
results = runner.run()

# 5. Compute metrics
metrics = MetricsCalculator().compute(results)
print(f"Failure rate: {metrics.failure_rate:.1%}")
print(f"MTTR: {metrics.mttr_ms:.0f}ms")
print(f"P99 latency: {metrics.p99_latency_ms:.0f}ms")
```

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
│   ├── tasks/                   # tool_use, rag, multi_agent task definitions
│   ├── runner.py                # Multi-run experiment orchestration
│   └── metrics.py               # MTTR, failure rate, P99 computation
└── agents/
    ├── langgraph_agent.py       # LangGraph ReAct agent
    ├── langchain_agent.py       # LangChain AgentExecutor
    └── autogen_agent.py         # AutoGen two-agent pipeline
```

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

| Framework | Status |
|-----------|--------|
| LangGraph | Full support |
| LangChain | Full support |
| AutoGen | Full support |

| LLM Backend | Install | Notes |
|-------------|---------|-------|
| Ollama + Llama 3.1 | `pip install langchain-ollama` | Local, zero API cost |
| OpenAI GPT-4o | `pip install langchain-openai` | Requires `OPENAI_API_KEY` |
| Anthropic Claude | `pip install langchain-anthropic` | Requires `ANTHROPIC_API_KEY` |

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v

# Skip slow/integration tests
pytest tests/ -v -m "not slow and not integration"
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

Phase 2 uses the corrected summary at `experiments/results_v2/` (not the original `experiments/results/combined_summary.json`, which contains known counting bugs). The paper's tables are derived from:
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
