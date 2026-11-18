# AgentProbe

[![CI](https://github.com/chhayankjain/agentprobe-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/chhayankjain/agentprobe-toolkit/actions/workflows/ci.yml)
[![CodeQL](https://github.com/chhayankjain/agentprobe-toolkit/actions/workflows/codeql.yml/badge.svg)](https://github.com/chhayankjain/agentprobe-toolkit/actions/workflows/codeql.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/agentprobe-toolkit.svg)](https://pypi.org/project/agentprobe-toolkit/)
[![GitHub release](https://img.shields.io/github/v/release/chhayankjain/agentprobe-toolkit)](https://github.com/chhayankjain/agentprobe-toolkit/releases)

**Test how your LLM agent handles tool failures, in pytest.**

Your agent's tools will time out, hit rate limits, and return garbage. AgentProbe wraps the tools your agent already uses, breaks them on purpose and deterministically, and checks what your agent did about it. The test fails unless every fault you declared **actually fired**, so an agent that skips the tool or answers from a cache can't look resilient. The examples and the demo run without a model or API key.

[Install](#install) · [Your first fault test](#your-first-fault-test) · [What a failure looks like](#what-a-failure-looks-like) · [Scenarios](#what-you-can-break) · [Assertions](#what-you-can-check) · [CI](#run-it-in-ci) · [Tutorial](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/examples/your_agent/README.md) · [For researchers](#for-researchers-the-benchmark-and-the-paper)

---

## Install

```bash
pip install "agentprobe-toolkit>=1.4" pytest
```

The testing API on this page (`agentprobe.testing`) needs version 1.4 or later.

You install `agentprobe-toolkit` and import `agentprobe`, the same way `pip install scikit-learn` gives you `import sklearn`. Python 3.11+.

## Your first fault test

Say `answer(question, tools)` is your agent and `TOOLS` maps tool names to the functions it calls. This test checks that when the payment service is down, the agent admits it instead of claiming success:

```python
from agentprobe.testing import probe

from my_agent import TOOLS, answer


def test_admits_failure_when_payment_service_is_down():
    with probe() as p:
        tools = p.wrap(TOOLS)                                   # probed copies; your tools are untouched
        p.on("payment_status").always_fail(with_="timeout")     # break one tool, every call
        result = p.run(lambda q: answer(q, tools), "Did order 42 go through?")
        result.assert_no_crash()
        result.assert_answer_not_contains("is confirmed")       # never claim unverified success
        result.assert_answer_contains("couldn't confirm")       # say so instead
        result.assert_not_called("send_receipt")                # no receipt for an unconfirmed payment
```

Run it with `pytest`. Nothing is mocked except the fault: your agent runs for real, with real tools, and AgentProbe only steps in at the tool boundary.

## What a failure looks like

Point the same test at an agent that swallows the error and reports success anyway, and pytest shows you exactly what happened:

```text
AssertionError: answer does not contain 'is confirmed': answer was 'Order 42 is confirmed and paid; receipt sent for order 42.'

Observed
  1. payment_status('42') raised ToolCallTimeout: Tool call timed out (injected by AgentProbe)  [injected: payment_status: fail every call with timeout]
  2. send_receipt('42') returned 'receipt sent for order 42'
  scenarios:
    [exercised] payment_status: fail every call with timeout (required; injected 1/open-ended, 1 call(s) to payment_status)
Asserted
  [fail] answer does not contain 'is confirmed': answer was 'Order 42 is confirmed and paid; receipt sent for order 42.'
Inferred (heuristics over the call log)
  fallbacks: [('payment_status', 'send_receipt')]
```

And an agent that never calls the tool (a cache, a hard-coded answer) fails even if every assertion would pass:

```text
AgentProbe gate FAILED: scenarios 0/1 required exercised, assertions 0/0 required passed
  - scenario 'payment_status: fail every call with timeout' not exercised: the agent never called 'payment_status' (tools called: none). It may have bypassed the tool or used a cached answer.
```

## What you can break

Scenarios are declared per tool with `p.on("tool_name")` and are deterministic: the same calls fail on every run.

| Scenario | Use it to test |
|---|---|
| `fail_first(n)` | Recovery: the first *n* calls fail, then the tool works |
| `fail_times(n).then_succeed()` | The same, spelled out |
| `always_fail()` | Outage: every call fails. Does the agent give up honestly? |
| `returns(value, times=1)` | Wrong, stale or incomplete data instead of an exception |
| `slow(ms=..., times=1)` | Latency, together with `p.run(..., timeout=...)` |
| `with_probability(p, seed=...)` | Random, seeded faults for experiments |

Faults: `"timeout"`, `"rate_limit"`, `"api_error"`, `"malformed_output"`, or any exception class or instance via `with_=`. Rules for one tool apply in order, so `fail_first(1).returns("stale", times=2)` fails call 1, returns stale data for calls 2 and 3, then lets calls through.

## What you can check

| Assertion | Passes when |
|---|---|
| `assert_answer_contains(text)` | The agent produced an answer containing *text* |
| `assert_answer_not_contains(text)` | The agent produced an answer **without** *text*. A crash or `None` never passes |
| `assert_no_crash()` | No exception escaped the agent |
| `assert_within(seconds)` | The run finished in time |
| `assert_called(tool)` / `assert_not_called(tool)` | The tool was or wasn't called; in-flight calls count |
| `assert_scenarios_exercised()` | Every required fault fired (also enforced automatically when the `with` block ends) |

The result keeps what was **observed** (every tool call with arguments, outcome and injected fault), what you **asserted** (pass, fail or unassessed) and what is **inferred** (`result.retries`, `result.fallbacks`) apart. `print(result.summary())` shows all three; `result.to_dict()` is versioned, JSON-safe output for CI artifacts.

## Works with your stack

- **Plain functions**, sync or `async`: pass a list, tuple or `{name: function}` dict to `p.wrap(...)`.
- **LangChain tools** (`@tool`, `StructuredTool`, `BaseTool` subclasses): wrapped as copies with the same schema.
- **LangGraph**: pass the wrapped tools to your graph. The prebuilt ReAct agent re-raises tool exceptions by default; use `ToolNode(tools, handle_tool_errors=True)` if your agent should see the error and recover.
- **Any agent callable** or object with `invoke()`: `p.run(agent, input)`.
- **Async agents and async tests:** `await p.arun(agent, input, timeout=...)`. At the deadline the probe stops injecting and requests cancellation; an agent that ignores it is reported as still running, never as passed.

What AgentProbe guarantees about your tools (never modified, cleanup on exceptions and cancellation, concurrency, agents that outlive their deadline) is in [docs/instrumentation.md](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/docs/instrumentation.md).

### Good to know before you wire in real tools

- **Real tools really run.** A call that gets a fault (`fail_*`, `always_fail`, `returns`) never reaches your tool, but every other call does, including calls delayed by `slow()`. If a tool sends email, charges a card or writes to a database, point it at a sandbox or a test double in your fault tests.
- **Reports are not redacted yet.** Summaries, assertion messages and `to_dict()` include tool arguments, return values and answers as they were. Keep secrets and customer data out of fault-test inputs, or don't upload the JSON as a CI artifact. Automatic redaction is planned ([#102](https://github.com/chhayankjain/agentprobe-toolkit/issues/102)).

## Run it in CI

Fault tests are ordinary pytest tests, so any CI runs them. A ready-made GitHub Actions workflow is in [`examples/your_agent/ci.yml`](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/examples/your_agent/ci.yml):

```yaml
- run: pip install "agentprobe-toolkit>=1.4" pytest
- run: pytest tests/ --junitxml=agentprobe-results.xml
```

The [10-minute tutorial](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/examples/your_agent/README.md) walks through a complete example agent, shows the same tests catching a deliberately broken version, and sets up CI. The [real-model example](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/examples/real_model_agent/README.md) runs the same kind of tests on a LangChain agent with a local Ollama model, and shows what they caught across models and prompts.

## Try the demo (no model, no key)

With AgentProbe installed as above:

```bash
python -m agentprobe.demo                      # rate-limit faults, p=0.3, 10 runs
python -m agentprobe.demo --fault timeout --p 0.5 --runs 20
python -m agentprobe.demo --json               # machine-readable report
```

The demo runs a small LangGraph agent on a scripted, deterministic model, injects seeded faults into its calculator tool, and prints the metrics, the per-call injection log with OpenTelemetry span ids, and each run's outcome.

## Get involved

- **Have a real agent?** Try AgentProbe on it and tell us where it hurt ([#68](https://github.com/chhayankjain/agentprobe-toolkit/issues/68)). That's the most valuable contribution there is.
- **Want to contribute code?** [Good first issues](https://github.com/chhayankjain/agentprobe-toolkit/labels/good%20first%20issue) are small and need no model or API key. Setup is in [CONTRIBUTING.md](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/CONTRIBUTING.md).
- **Where it's going:** [ROADMAP.md](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/ROADMAP.md). Next up: a pytest plugin, secret redaction in reports, tool-call argument and order checks, and a fault coverage report.
- **Questions and ideas:** [Discussions](https://github.com/chhayankjain/agentprobe-toolkit/discussions).

---

## For researchers: the benchmark and the paper

AgentProbe began as the toolkit behind the paper *"AgentProbe: Systematic Fault Injection and Observability for Evaluating the Reliability of LLM-Based Intelligent Agents"* (submitted to MAKE, MDPI, 2026). The benchmark side runs agents many times under seeded, probabilistic faults and reports failure rate, recovery rate, MTTR and latency percentiles across LangGraph, LangChain and AutoGen. Everything below documents that toolkit and how to reproduce the paper.

```bash
# From a clone, for development or to run the paper's experiments
git clone https://github.com/chhayankjain/agentprobe-toolkit.git && cd agentprobe-toolkit
pip install -e ".[dev,plotting,ollama]"

# The AutoGen adapter targets pyautogen 0.1.14 and needs openai<1.0, which
# conflicts with langchain-openai. Install it in its own environment:
#   pip install "agentprobe-toolkit[ollama,autogen]"
```

### Failure taxonomy

AgentProbe organises agent failures into five categories. Only the first has been exercised in a published experiment; the status column says exactly what exists for each.

| # | Category | Examples | Status in this repo |
|---|----------|----------|---------------------|
| 1 | **Tool Call** | API timeouts, malformed output, rate limits (HTTP 429), server errors (5xx) | Injector implemented, **used in all paper experiments** (`injector/tool_failure.py`) |
| 2 | **Orchestration** | Infinite loops, wrong branching, state corruption, dropped handoffs | Injector implemented and unit-tested, not yet used in an experiment (`injector/orchestration_failure.py`) |
| 3 | **Context** | Context window overflow, lost history, hallucinated history | Injector implemented and unit-tested, not yet used in an experiment (`injector/context_failure.py`) |
| 4 | **Latency** | P99 spikes, cascading slowdowns, jitter | Injector implemented; latency in the paper is a by-product of tool faults, not independently injected (`injector/latency_failure.py`) |
| 5 | **Consistency** | Non-deterministic outputs, divergent multi-agent state | Detected only, via a Jaccard-similarity classifier; nothing is injected (`observer/classifier.py`) |

Note that the current "malformed output" injector raises an exception at the tool boundary rather than returning corrupted data to the model, so it tests the agent's exception path, not its parsing. See issue [#28](https://github.com/chhayankjain/agentprobe-toolkit/issues/28).

### Benchmark quick start (Ollama)

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

### Benchmark API

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

#### Probabilistic injectors (lower-level API)

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

#### Two prompt sets

- **Paper prompts (default).** `BenchmarkRunner` cycles the five prompts per task in `agentprobe.benchmark.runner.PAPER_PROMPTS`. Every batch in the paper used these.
- **Graded corpora.** `agentprobe.benchmark.tasks` holds 25 tool-use, 15 RAG, and 15 multi-agent queries with keyword graders. They have not been used in a published experiment. Run them with `task_fn=corpus_task_fn("tool_use")` or `run_benchmarks.py --query-set corpus`.

#### Metric definitions

- **Failure**: the agent invocation raised an exception that reached the harness. A wrong-but-exception-free answer is *not* a failure here; use the grader for that.
- **Recovery**: the harness re-invoked the agent once (injector still active) and that call succeeded. This is a harness-level retry, not agent-internal self-repair.
- **MTTR**: mean latency of successful re-invocations, in ms.
- **P50/P90/P99**: ceiling nearest-rank percentiles of per-run latency. For n < 100, P99 equals the sample maximum. Injected timeouts sleep for real, so latency under the timeout condition includes the injected delay.
- **Recovery rate** is `None` when a condition had no failures.

### Architecture

```mermaid
flowchart LR
    SCRIPT["experiment script<br/>or your code"] --> RUNNER["BenchmarkRunner<br/>N runs · timeout · 1 retry"]
    RUNNER --> ADAPTER["Adapter<br/>LangGraph · LangChain · AutoGen"]
    ADAPTER <--> LLM["LLM<br/>Ollama / OpenAI-compatible"]
    ADAPTER --> INJ["ToolFailureInjector<br/>seeded · p per call"]
    INJ -- "pass (1-p)" --> TOOL["tool fn"]
    INJ -- "inject (p)" --> FAULT["raise timeout / 429 / 5xx / malformed"]
    RUNNER --> METRICS["MetricsCalculator<br/>failure & recovery rate · MTTR · P99"]
    RUNNER & ADAPTER & INJ --> OTEL["OpenTelemetry spans<br/>console · Jaeger"]
    INJ --> LOG["injection log<br/>args · result · span_id"]
```

Full diagrams (per-trial sequence, injector internals, data flow, observability stack, CI/release) are in [ARCHITECTURE.md](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/ARCHITECTURE.md).

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

See [ARCHITECTURE.md](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/ARCHITECTURE.md) for design decisions.

### Observability stack

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

### Supported frameworks and LLM backends

| Framework | Adapter | Validated in paper | Notes |
|-----------|---------|--------------------|-------|
| LangGraph | `LangGraphToolAgent` | Yes (Llama 3.1, Mistral 7B, Qwen2.5 7B) | Hand-built ReAct graph |
| LangChain | `LangChainToolAgent` | Yes (same models) | Uses `create_agent`, which compiles to LangGraph; shares an execution path with the LangGraph adapter, so results converge |
| AutoGen | `AutoGenToolAgent` | Yes, with caveats | Targets **pyautogen 0.1.14** through a `functions`→`tools` compatibility shim. Phase 1 found it silently bypassed tools; Phase 2 patched that. Modern `ag2` / `autogen-agentchat` 0.4+ is not yet supported ([roadmap](https://github.com/chhayankjain/agentprobe-toolkit/issues/38)) |

| LLM Backend | `--provider` | Install extra | Validated in paper | Notes |
|-------------|--------------|---------------|--------------------|-------|
| Ollama (Llama 3.1 8B, Mistral 7B, Qwen2.5 7B, Q4_K_M) | `ollama` (default) | `.[ollama]` | Yes | Local, zero API cost. Every number in the paper comes from this backend |
| Any OpenAI-compatible server (vLLM, LM Studio, llama.cpp, ...) | `openai-compatible --base-url …` | `.[openai]` | **No** | |
| OpenAI | `openai` | `.[openai]` | **No** | `OPENAI_API_KEY` |
| Groq | `groq` | `.[openai]` | **No** | `GROQ_API_KEY`; free tier is token-per-day limited, use `--rpm` |
| OpenRouter | `openrouter` | `.[openai]` | **No** | `OPENROUTER_API_KEY`; `:free` models share a low daily cap |
| Cerebras | `cerebras` | `.[openai]` | **No** | `CEREBRAS_API_KEY` |
| Mistral | `mistral` | `.[openai]` | **No** | `MISTRAL_API_KEY` |
| GitHub Models | `github` | `.[openai]` | **No** | `GITHUB_TOKEN` with the `models` permission |
| Google AI Studio (Gemini), AutoGen adapter | `gemini` | none | **No** | `GEMINI_API_KEY`, via Google's OpenAI-compatible endpoint; works with pyautogen's legacy client |
| Google AI Studio (Gemini), LangGraph/LangChain adapters | `google` | `.[google]` | **No** | `GEMINI_API_KEY`, native `langchain-google-genai`. Required for Gemini 3 models: their tool calls carry thought signatures that the OpenAI-compatible endpoint drops on the second turn (HTTP 400) |
| Anthropic (Claude) | `anthropic` | `.[anthropic]` | **No** | `ANTHROPIC_API_KEY`; LangGraph/LangChain adapters only |

#### Hosted backends

Every experiment script (`run_experiment_v2.py`, `run_qualification.py`, `run_benchmarks.py`, `smoke_test_model.py`, `autogen_diagnostic.py`) takes the same flags, defined in `agentprobe.llm`:

```bash
# Default: identical to the paper's runs
python experiments/run_experiment_v2.py --model llama3.1

# Groq free tier, throttled to stay under the daily token budget
export GROQ_API_KEY=...
python experiments/smoke_test_model.py --provider groq --model llama-3.3-70b-versatile --frameworks langgraph
python experiments/run_experiment_v2.py --provider groq --model llama-3.3-70b-versatile --rpm 25 \
    --frameworks langgraph langchain

# Google AI Studio: native client for LangGraph/LangChain, OpenAI-compatible endpoint for AutoGen
export GEMINI_API_KEY=...
python experiments/run_experiment_v2.py --provider google --model gemini-flash-lite-latest --rpm 10 --frameworks langgraph langchain
python experiments/run_experiment_v2.py --provider gemini --model gemini-flash-lite-latest --rpm 10 --frameworks autogen

# Anything OpenAI-compatible
python experiments/run_experiment_v2.py --provider openai-compatible --base-url http://localhost:8000/v1 \
    --api-key-env VLLM_KEY --model my-model
```

Flags: `--provider`, `--model`, `--base-url`, `--api-key-env`, `--rpm` (client-side throttle), `--max-retries` (client retries on 429/5xx/timeouts, default 6), `--llm-timeout`.

Three things the harness does for hosted runs:

- **Backend errors are not agent failures.** A provider-side 429, 5xx, or connection error that survives client retries is re-dispatched by the runner (`backend_error_retries`, default 2, with backoff), and if it still fails the trial is recorded as `status="backend_error"` with category `infrastructure` and **excluded from every metric**. It is never confused with an *injected* `ToolCallRateLimited`. Conditions with backend errors are marked incomplete.
- **Exact backend is recorded.** `experiment_metadata.json` gains `llm_provider`, `model_id`, `llm_base_url`, `llm_api_key_env` (the variable *name*, never the key), `llm_requests_per_minute`, so a batch says which hosted model produced it. `--resume` refuses a batch whose provider or endpoint changed.
- **Backend errors are recognised at the model call, not by exception type.** A tool that fails with the same transport error (`httpx.ReadTimeout`, a 503 from a real service) is a measured agent failure. If you write your own agent, wrap each model call in `agentprobe.llm.llm_call_boundary()` to get the same separation.
- **AutoGen runs one tool call per turn.** A turn that requests several parallel calls is recorded as a `harness_error` and excluded, rather than silently running only the first.
- **AutoGen works with any OpenAI-compatible provider** through `api_base`, so hosted AutoGen runs need no extra packages. LangGraph/LangChain hosted runs need `langchain-openai` (`openai>=1`), which conflicts with the `[autogen]` extra (`openai<1`): use a separate environment for those.

Hosted models change silently and are not seed-reproducible; treat hosted batches as a snapshot, not a replication of the paper.

Free-tier notes from testing (November 2026): Google AI Studio's `gemini-3-flash-preview` allows 20 requests per day per project, enough for a smoke test but not a batch; the `*-flash-lite` models have separate, larger quotas, shown only inside AI Studio. Google now issues keys starting with `AQ.` as well as `AIza`. Groq's free tier is capped by tokens per day. Check the per-model quota in the provider console before launching a 450-run batch.

### Reproducing the paper's results

The paper reports a two-phase study using Llama 3.1 8B (Q4_K_M) via Ollama, with injection seed 42 throughout.

#### Phase 1 — Discovery (1,050 runs)

Phase 1 uses `run_benchmarks.py` with 30 runs for LangGraph/LangChain and 10 for AutoGen across 3 task types and 5 fault conditions. This phase discovered that AutoGen never invokes tools due to a legacy API-parameter mismatch.

> **Historical reproduction note:** The current codebase includes the patched AutoGen adapter (`_patch_openai_compat()`). To reproduce the original Phase 1 bypass behavior, use the pre-patch source at tag `v1.0.0`. Running Phase 1 commands with the current code will produce patched AutoGen behavior (tools are called), not the historical bypass.

```bash
# LangGraph / LangChain (30 runs per condition)
python experiments/run_benchmarks.py \
    --frameworks langgraph langchain \
    --tasks tool_use rag multi_agent \
    --failure-types none timeout malformed_output rate_limit api_error \
    --n-runs 30 --provider ollama --model llama3.1 --save-results

# AutoGen (10 runs per condition)
python experiments/run_benchmarks.py \
    --frameworks autogen \
    --tasks tool_use rag multi_agent \
    --failure-types none timeout malformed_output rate_limit api_error \
    --n-runs 10 --provider ollama --model llama3.1 --save-results
```

#### Phase 2 — Patched Validation (450 runs)

Phase 2 uses `run_experiment_v2.py` with the patched AutoGen adapter (proper `tool_name=` recording, `_patch_openai_compat()` for the functions/tools protocol gap). Runs 30 trials per condition across 3 frameworks × 5 fault conditions on tool-use tasks only, with matched p=0.20 injection probability.

```bash
python experiments/run_experiment_v2.py \
    --frameworks langgraph langchain autogen \
    --model llama3.1 --seed 42
```

#### Qualification Experiment (12 trials)

A small, targeted validation with deterministic calculator tasks and guaranteed fault injection (p=1.0). Produces a self-contained JSON artifact with inputs, outputs, tool events, injection log entries (with args/returns), and linked OTel trace span IDs.

```bash
python experiments/run_qualification.py \
    --frameworks langgraph langchain autogen \
    --model llama3.1
```

#### Cross-Model Validation

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

#### AutoGen Diagnostic (Direct Client vs Shim)

A diagnostic comparing the AutoGen adapter pipeline against a direct OpenAI-compatible client on the same three deterministic tasks. The compared paths differ in message format, stopping policy, and temperature handling; results are consistent with the adapter layer being a major contributor to low answer-string presence but do not isolate a single cause:

```bash
# Llama 3.1
python experiments/autogen_diagnostic.py --model llama3.1

# Qwen2.5 7B
python experiments/autogen_diagnostic.py --model qwen2.5:7b
```

Diagnostic artifacts are saved to `experiments/results_v2/diagnostic_*/`.

#### Analysis

Phase 2 uses the corrected summaries under `experiments/results_v2/`. The original Phase 1 output (now at `experiments/archive_v1/results/combined_summary.json`) contains known counting bugs and is kept only for the historical record. The paper's tables are derived from:
- **Phase 2 batch (Llama 3.1):** `experiments/results_v2/20261111_113514/`
- **Phase 2 batch (Mistral 7B):** `experiments/results_v2/20261111_233435/`
- **Phase 2 batch (Qwen2.5 7B LG/LC):** `experiments/results_v2/20261112_040156/`
- **Phase 2 batch (Qwen2.5 7B AutoGen rerun):** `experiments/results_v2/20261112_063451/` (corrected extractor)
- **Qualification (Llama 3.1):** `experiments/results_v2/qualification_20261111_211913/` (12 trials, 10 passed)
- **Qualification (Mistral 7B):** `experiments/results_v2/qualification_20261111_233116/` (12 trials, 8 passed)
- **Qualification (Qwen2.5 7B):** `experiments/results_v2/qualification_20261111_233320/` (12 trials, 10 passed)

> **Source map:** The authoritative mapping from each paper table cell to its batch directory is in [`experiments/results_v2/paper_source_map.json`](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/experiments/results_v2/paper_source_map.json). The Qwen2.5 column combines two batches: LangGraph/LangChain from `20261112_040156` and AutoGen from `20261112_063451` (corrected output extractor). Llama and Mistral use their single original batches. The `analyze_cross_model.py` script does not automatically merge these; use the source map to select the correct batch per framework.

> **Model digests:** All three models use Q4_K_M quantization. Ollama digest prefixes: Llama `46e0c10c039e`, Mistral `6577803aa9a0`, Qwen `845dbda0ea48`. The `experiment_metadata.json` files in historical batches record `model_digest: unknown` because digest capture was added after the experiments ran; the digests listed here were identified post-hoc from the same Ollama installation.

## Development

```bash
git clone https://github.com/chhayankjain/agentprobe-toolkit.git && cd agentprobe-toolkit
pip install -e ".[dev,plotting]"                     # no model or API key needed
pytest tests/ examples/your_agent -m "not slow and not integration"
ruff check agentprobe/ tests/ examples/ && ruff format --check agentprobe/ tests/ examples/
```

See [CONTRIBUTING.md](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/CONTRIBUTING.md) for the workflow and [ARCHITECTURE.md](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/ARCHITECTURE.md) for how the code fits together.

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

MIT — see [LICENSE](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/LICENSE).
