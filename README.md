# AgentProbe

**Reliability & Observability Toolkit for Production Agentic LLM Systems**

AgentProbe is an open source benchmarking toolkit that injects real-world failure modes into LangGraph, LangChain, and AutoGen agents, traces execution via OpenTelemetry, and measures reliability metrics (MTTR, failure rate, P99 latency) across frameworks and workloads.

Companion to the paper: *"Failure Modes and Observability Patterns in Production Agentic LLM Systems: A Systematic Study"* (arXiv, 2026).

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

# Start Ollama with Llama-3
ollama pull llama3

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
llm = ChatOllama(model="llama3", temperature=0.0)
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
| Ollama + Llama-3 | `pip install langchain-ollama` | Local, zero API cost |
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

```bash
# Full benchmark matrix (50 runs per config, ~2-3h with Ollama)
python experiments/run_benchmarks.py \
    --frameworks langgraph langchain autogen \
    --tasks tool_use rag multi_agent \
    --failure-types none timeout malformed_output rate_limit api_error \
    --n-runs 50 \
    --llm ollama \
    --model llama3 \
    --save-results

# Results saved to experiments/results/
```

## Citation

```bibtex
@misc{jain2026agentprobe,
  title={Failure Modes and Observability Patterns in Production Agentic LLM Systems: A Systematic Study},
  author={Jain, Chhayank},
  year={2026},
  eprint={PLACEHOLDER},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

## License

MIT — see [LICENSE](LICENSE).
