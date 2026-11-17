# Architecture

This document explains AgentProbe's internal design to help contributors navigate the codebase. Diagrams are [Mermaid](https://mermaid.js.org/) and render on GitHub; edit them in place when the code changes.

## 1. System overview

Four library layers plus the experiment harness that drives them. Only `agentprobe/` ships in the PyPI package; `experiments/` is the paper's harness and stays in the repo.

```mermaid
flowchart TB
    subgraph EXP["experiments/  (paper harness, not packaged)"]
        direction LR
        RUNv2["run_experiment_v2.py<br/>Phase 2 matrix + checkpoints"]
        QUAL["run_qualification.py<br/>p=1.0 trace-linked validation"]
        DIAG["autogen_diagnostic.py<br/>direct client vs shim"]
        AN["analyze_*.py<br/>generate_paper_figures.py"]
    end

    subgraph PKG["agentprobe/  (PyPI: agentprobe-toolkit)"]
        direction TB
        subgraph BENCH["benchmark/"]
            RUNNER["BenchmarkRunner<br/>N runs · timeout thread · 1 retry"]
            METRICS["MetricsCalculator<br/>failure/recovery rate · MTTR · P50/P90/P99"]
            GRADER["ResponseGrader<br/>keyword / exact-answer grading"]
            STATS["statistics.py<br/>Wilson CI · Cohen's d/h"]
            TASKS["PAPER_PROMPTS (default)<br/>tasks/ corpora (55 queries)"]
            META["metadata.py<br/>git · pip freeze · ollama · hardware"]
        end
        subgraph AGENTS["agents/  (BaseAgent protocol)"]
            LG["LangGraphToolAgent<br/>hand-built ReAct graph"]
            LC["LangChainToolAgent<br/>create_agent → LangGraph"]
            AG["AutoGenToolAgent<br/>pyautogen 0.1.14 + functions→tools shim"]
        end
        subgraph INJ["injector/"]
            TOOLINJ["ToolFailureInjector<br/>timeout · malformed · 429 · 5xx<br/>seeded RNG · per-call log"]
            OTHERINJ["Orchestration · Context · Latency<br/>injectors (implemented, unvalidated)"]
        end
        subgraph OBS["observer/"]
            TRACER["AgentProbeTracer<br/>OTel provider · console + OTLP"]
            CLS["FailureClassifier<br/>exception → taxonomy category"]
            PROM["PrometheusExporter<br/>/metrics on :8000"]
        end
    end

    subgraph EXT["external"]
        LLM["LLM backend via llm.py<br/>Ollama (paper) · Groq · OpenRouter · Gemini ·<br/>any OpenAI-compatible · Anthropic"]
        OTELB["Jaeger / Tempo<br/>OTLP :4317"]
        PROMS["Prometheus → Grafana"]
    end

    RUNv2 & QUAL & DIAG --> RUNNER
    RUNNER --> LG & LC & AG
    RUNNER --> METRICS & GRADER & META
    METRICS --> STATS
    TASKS --> RUNNER
    LG & LC & AG --> TOOLINJ
    LG & LC & AG --> LLM
    TOOLINJ & RUNNER & LG & LC & AG --> TRACER
    RUNNER --> CLS
    TRACER --> OTELB
    PROM --> PROMS
    RUNv2 --> AN
```

**Reading the diagram.** The experiment scripts never touch the frameworks directly: they build an adapter, hand it to `BenchmarkRunner`, and read back `RunResult` records. Each adapter creates its own `ToolFailureInjector` and wraps its tools with it, so injection happens identically regardless of framework. Every layer emits OpenTelemetry spans through the shared tracer.

## 2. One trial, end to end

What happens inside `BenchmarkRunner._run_single()` for a single run under injection. Span names are the real ones.

```mermaid
sequenceDiagram
    autonumber
    participant R as BenchmarkRunner
    participant W as worker thread<br/>(timeout guard)
    participant A as Adapter<br/>(LangGraph shown)
    participant M as LLM
    participant I as ToolFailureInjector
    participant T as tool fn<br/>(calculator / web_search)
    participant C as FailureClassifier

    R->>R: open span benchmark.run.{id}
    R->>W: submit invoke(task_input), deadline = timeout_per_run_seconds
    W->>A: invoke({"input": prompt})
    A->>A: open span langgraph.agent.invoke
    loop ReAct until no tool_calls or recursion_limit
        A->>M: chat(messages + tool schemas)  [span langgraph.node.agent]
        M-->>A: AIMessage(tool_calls=[calculator(expr)])
        A->>I: patched _run(expr)
        I->>I: call_index += 1 · draw rng.random(), inject if below p
        alt inject
            I->>I: open span tool.call.calculator<br/>failure.injected=true, failure.type=…
            Note over I: TIMEOUT also sleeps timeout_seconds
            I-->>A: raise ToolCallTimeout / ToolCallRateLimited / …
        else pass through
            I->>T: real call
            T-->>I: result
            I-->>A: result  (span status OK)
        end
        I->>I: append InjectionResult(args, return/exception, span_id)
    end
    A-->>W: {"output", "latency_ms", "framework"}
    W-->>R: result or exception

    alt exception reached the harness
        R->>C: classify_exception(exc) → category
        R->>W: recovery: invoke(task_input) once more (injector still active)
        W-->>R: recovered output or exception
        R->>R: RunResult(success=False, recovered=?, recovery_latency_ms)
    else returned normally
        R->>R: RunResult(success=True, output)
    else deadline missed
        R->>R: RunResult(status="completed", _HarnessTimeout) · mark invocation_hung<br/>remaining trials in condition → status="skipped"
    end
```

**Three things this diagram makes explicit.**

- *Injection is probabilistic per tool call, not per run.* `call_index` and the RNG are condition-wide, so `injector_calls` and `injector_injections` in the output are totals for the whole condition, never per trial.
- *Recovery is a harness-level retry.* The runner re-invokes the whole agent once with the injector still active. MTTR is the latency of that second invocation. It is not agent-internal self-repair.
- *A hung invocation poisons the rest of the condition.* Python threads cannot be killed, so after a deadline miss the runner skips the remaining trials rather than risk two concurrent invocations on one agent. Skipped trials are excluded from every metric.

## 3. Fault injection internals

```mermaid
flowchart LR
    subgraph CFG["ToolFailureConfig (dataclass)"]
        direction TB
        FT["failure_type<br/>NONE · TIMEOUT · MALFORMED_OUTPUT<br/>RATE_LIMIT · API_ERROR"]
        P["failure_probability p"]
        SEED["seed → random.Random(seed)"]
        EXTRA["timeout_seconds · retry_after_seconds<br/>error_code · malformed_payload"]
    end

    subgraph ENTRY["entry points"]
        WRAP["@injector.wrap(fn)"]
        CALL["injector.call(fn, *args, tool_name=…)"]
        LCT["injector.inject_langchain_tool(tool)<br/>monkey-patches tool._run / _arun in place"]
    end

    subgraph EXEC["_execute_sync / _execute_async"]
        direction TB
        IDX["call_index += 1"]
        DEC{"draw below p<br/>and type is not NONE?"}
        SPAN["open span tool.call.{name}<br/>attrs: tool.name, tool.call_index,<br/>failure.injected, failure.type"]
        RAISE["build exception<br/>(TIMEOUT sleeps first)"]
        REAL["fn(*args, **kwargs)"]
        LOG["InjectionResult → injection_log<br/>args · kwargs · return/exception<br/>latency_ms · span_id"]
    end

    CFG --> EXEC
    WRAP & CALL & LCT --> IDX --> DEC
    DEC -- yes --> SPAN --> RAISE --> LOG
    DEC -- no --> SPAN --> REAL --> LOG
    LOG --> SUM["summary()<br/>total_calls · injected_failures<br/>injection_rate · failures_by_type"]
```

All four fault types currently **raise** at the tool boundary. "Malformed output" therefore exercises the agent's exception path, not its parsing; returning corrupted-but-valid data is tracked in issue #28. Because every record carries the OTel `span_id`, a qualification run can prove that a specific injected fault appears in a specific trace.

## 4. Experiment data flow

How a batch directory is produced and consumed. File names are exact.

```mermaid
flowchart LR
    CLI["run_experiment_v2.py<br/>--frameworks --model --seed<br/>[--resume BATCH] [--pilot]"]

    subgraph BATCH["experiments/results_v2/{YYYYMMDD_HHMMSS}/"]
        direction TB
        COND["{framework}_{task}_{fault}.json / .csv<br/>one EnhancedRunRecord per trial"]
        CKPT["checkpoint.json<br/>structured (framework, task, fault) triples<br/>validated on --resume"]
        SUMM["combined_summary.json<br/>AgentMetrics per condition"]
        METAJ["experiment_metadata.json<br/>model · seed · git hash · dirty flag<br/>package versions · hardware"]
        MANI["manifest.json<br/>file inventory"]
        SNAP["source_snapshot/<br/>runner.py · grader.py · metrics.py<br/>pip_freeze.txt · uncommitted.patch"]
    end

    CLI -->|per condition, after validation| COND
    CLI -->|after each condition| CKPT
    CLI -->|end of batch| SUMM & METAJ & MANI & SNAP

    QUALCLI["run_qualification.py"] --> QUALD["results_v2/qualification_{ts}/<br/>qualification_results.json<br/>inputs · outputs · tool events ·<br/>injection log · linked span IDs"]
    DIAGCLI["autogen_diagnostic.py"] --> DIAGD["results_v2/diagnostic_{ts}/<br/>diagnostic_{model}.json"]

    COND & SUMM --> XM["analyze_cross_model.py<br/>--batches … --models …"]
    COND & SUMM --> FIG["generate_paper_figures.py"]
    SUMM --> AR["analyze_results.py"]
    SRCMAP["paper_source_map.json<br/>table cell → batch dir"] -.authoritative mapping.-> XM & FIG

    FIG --> PLOTS["experiments/plots/*.pdf"]
    PLOTS -->|symlink paper/plots| TEX["paper/main.tex<br/>graphicspath"]
    XM & AR --> TABLES["paper tables"]
```

The Qwen2.5 column in the paper combines two batches (LangGraph/LangChain from one, AutoGen from a rerun with the corrected output extractor). `paper_source_map.json` is the only authoritative statement of which batch feeds which table cell; the analysis scripts do not merge batches on their own.

## 5. Observability stack

What `docker compose up -d` starts and how spans and metrics reach it.

```mermaid
flowchart LR
    subgraph HOST["host (benchmark process)"]
        TR["AgentProbeTracer<br/>TracerProvider + Resource(service.name)"]
        CON["ConsoleSpanExporter<br/>(SimpleSpanProcessor)<br/>off with AGENTPROBE_NO_CONSOLE_TRACE=1"]
        OTLP["OTLPSpanExporter gRPC<br/>(BatchSpanProcessor)<br/>AGENTPROBE_OTLP_ENDPOINT"]
        PE["PrometheusExporter<br/>start_http_server(8000)<br/>agentprobe_tool_call_total · _tool_failure_total<br/>_latency_seconds · _mttr_seconds · _failure_rate"]
    end

    subgraph DC["docker compose"]
        J["jaeger all-in-one :16686<br/>OTLP gRPC :4317 · HTTP :4318"]
        P["prometheus :9090<br/>scrapes host.docker.internal:8000 /metrics every 15 s"]
        G["grafana :3000<br/>provisioned Prometheus datasource"]
    end

    TR --> CON
    TR --> OTLP -->|:4317| J
    PE -->|scrape| P --> G
```

Spans form a tree per trial: `benchmark.run.{id}` → `{framework}.agent.invoke` → `langgraph.node.agent` (per LLM call) and `tool.call.{name}` (per tool call, with `failure.injected` and `failure.type` attributes). The classifier can also run offline over finished spans via `FailureClassifier.classify_spans()`.

## 6. CI and release pipeline

```mermaid
flowchart LR
    subgraph PR["on pull_request / push to main  (ci.yml)"]
        direction TB
        T["test · Python 3.11 / 3.12 / 3.13 / 3.14<br/>pytest -m 'not slow and not integration'"]
        L["lint · ruff check + ruff format --check"]
        TY["typecheck · mypy (advisory)"]
        B["build · python -m build · twine check --strict<br/>install wheel in clean venv · import agentprobe"]
        CQ["codeql.yml · weekly + on PR"]
    end

    subgraph REL["release.yml"]
        direction TB
        TAG["git push origin vX.Y.Z"] --> BLD["build job<br/>tag must equal pyproject version<br/>twine check · wheel smoke test"]
        BLD --> GATE{"environment: pypi<br/>required reviewer approves"}
        GATE --> PUB["pypa/gh-action-pypi-publish<br/>Trusted Publishing (OIDC, no tokens)"]
        PUB --> GHR["GitHub release<br/>generated notes + dist/* attached"]
        DISP["Run workflow (manual)"] --> BLD2["build job"] --> TPUB["publish to TestPyPI<br/>environment: testpypi"]
    end

    PR -. green checks required .-> TAG
    DEP["Dependabot<br/>github-actions: weekly, grouped<br/>pip: security advisories only"] -.-> PR
```

The version lives in three files (`pyproject.toml`, `agentprobe/__init__.py`, `CITATION.cff`) and `tests/test_public_api.py` fails if they disagree, so a mismatched tag cannot get past the build job.

## Core modules

### `agentprobe/agents/` — framework adapters

Each adapter wraps a specific agent framework behind the `BaseAgent` protocol in `agents/base.py`:

```python
framework: str
invoke(input: dict) -> dict      # input["input"] is the query; returns {"output", "latency_ms", "framework", ...}
reset() -> None                  # clear conversation state (never the injector)
get_injector_summary() -> dict   # condition-wide injection stats
```

All adapters accept the same tools (calculator, web_search) and the same LLM configuration. The adapter translates between the framework's native API and AgentProbe's common interface.

`LangChainToolAgent` is built on `langchain.agents.create_agent`, which compiles to a LangGraph graph internally. The two adapters therefore share an execution path, which is why the paper reports identical LangGraph/LangChain failure patterns. `AutoGenToolAgent` targets `pyautogen 0.1.14` and installs a one-time monkey-patch that converts the legacy `functions` request format to `tools` and the `tool_calls` response back to `function_call`, because Ollama silently ignores the legacy format (the Phase 1 "tool bypass" finding).

**Adding a new framework:** write a class that satisfies `BaseAgent` (it is a structural `Protocol`, no subclassing needed), wrap your tools with `ToolFailureInjector`, and register the class in `build_agent()` in `experiments/run_experiment_v2.py`.

### `agentprobe/injector/` — fault injectors

`ToolFailureInjector` is the one used in every experiment (Section 3). `OrchestrationFailureInjector`, `ContextFailureInjector`, and `LatencyFailureInjector` are implemented and unit-tested but have not been used in a published experiment (issues #24, #25).

**Key design choice:** injectors are stateful (call count, RNG, log) but the RNG is seeded, so the injection sequence is reproducible regardless of framework-internal randomness.

### `agentprobe/benchmark/` — experiment infrastructure

```
runner.py      — BenchmarkRunner, RunResult, BenchmarkConfig, PAPER_PROMPTS / paper_task_fn
metrics.py     — MetricsCalculator → AgentMetrics (failure/recovery rate, MTTR, percentiles)
statistics.py  — Wilson CIs, Cohen's d/h, comparison reports
grader.py      — ResponseGrader: keyword and exact-answer grading, silent-failure detection
tasks/         — graded query corpora (25 tool_use, 15 rag, 15 multi_agent); NOT the paper's prompts
metadata.py    — ExperimentMetadata capture (model, seed, git commit, pip freeze, hardware)
```

**Two prompt sets.** The runner's default `task_fn` is `paper_task_fn`, which cycles the five prompts per task in `runner.PAPER_PROMPTS`. Every batch reported in the paper used this set. The larger graded corpora are available via `corpus_task_fn(task)` but have not been used in a published experiment.

**Metric definitions.** "Failure" means the agent invocation raised an exception that reached the harness. "Recovery" means the harness's single re-invocation succeeded. MTTR is the mean latency of successful re-invocations. P99 is ceiling nearest-rank, so for n < 100 it equals the sample maximum. `recovery_rate` is `None` when a condition had no failures.

### `agentprobe/llm.py` — provider-agnostic backends

`LLMSpec` describes a backend (provider preset, exact model id, endpoint, key *variable*, throttle, retries); `build_chat_model()` returns the LangChain model for the LangGraph/LangChain adapters and `autogen_llm_config()` the `llm_config` for AutoGen. Every experiment script takes the same `--provider …` flags via `add_llm_args()`; the default is the paper's local Ollama. `is_llm_backend_error()` is what lets the runner keep provider-side 429/5xx events (recorded as `status="backend_error"`, excluded from metrics) apart from *injected* tool faults.

### `agentprobe/observer/` — observability

```
tracer.py      — AgentProbeTracer: OTel provider setup; get_tracer() for library modules
classifier.py  — FailureClassifier: exception name / span attributes → 5-category taxonomy; Jaccard consistency check
dashboard.py   — PrometheusExporter: counters, gauges, histogram on /metrics
```

The classifier is rule-based and is used by the runner to label `failure_category`; it is not used for the paper's headline metrics.

## Key design decisions

1. **Adapters, not plugins.** Each framework adapter is a concrete class. Simple and debuggable at the cost of a code change to add a framework.
2. **Injection at the tool boundary.** Faults wrap tool functions, not LLM responses or framework internals, so injection behaves identically across frameworks.
3. **Seeded RNG per injector, condition-wide state.** Reproducible sequences; totals are per condition.
4. **Checkpoint-based resume with validation.** The checkpoint stores structured `(framework, task, fault_type)` triples, and resume refuses to continue if record counts, run IDs, or metadata (model, seed, n_runs) disagree with what is on disk.
5. **Metrics are pure computation.** `MetricsCalculator` takes a list of run results and returns metrics. No side effects, easy to test.
6. **Library and harness are separate.** Only `agentprobe/` is packaged; experiment scripts import it like any user would.

## Testing

```
tests/
├── test_public_api.py             — public exports, BaseAgent protocol, config/injector compatibility, version sync
├── test_tool_failure.py           — injection probability and fault types
├── test_benchmark_runner.py       — runner orchestration, timeout handling
├── test_metrics.py                — metrics computation edge cases
├── test_experiment_state.py       — checkpoint validation and resume safety
├── test_pilot_hung.py             — hung invocation detection
├── test_qualification_checker.py  — answer validation logic
├── test_metadata.py               — metadata capture
└── ...                            — classifier, grader, statistics, tasks, tracer, other injectors
```

Tests use mocked agents and tools — no LLM calls required. Run with `pytest tests/ -v`.
