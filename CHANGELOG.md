# Changelog

## [Unreleased]

### Added
- Case study (`docs/case-studies/agents-from-scratch.md`) and example (`examples/third_party_agent/`): fault tests on a third-party LangGraph agent without editing its code; found the unhandled tool exception reported upstream as langchain-ai/agents-from-scratch#70
- `examples/real_model_agent/`: fault tests on a real LLM agent (LangChain `create_agent` on local Ollama). Recovery is checked on the call log (right order, right arguments, confirmation before the receipt), not on wording. Includes `run_matrix.py` and the saved evidence for its results table across two models, two prompts and two tool-error messages. Live tests are marked `integration`; CI compiles the example and runs its offline call-log checks

## [1.4.0] - 2026-11-18

Test the agent you already have: wrap its tools, declare deterministic faults, and fail the test unless every fault actually fired. See the README and the tutorial in `examples/your_agent/`.

### Maintenance
- Include the typing marker and documentation in distributions; wheel checks now exercise the public probe API and offline demo, and source-archive checks run the tutorial tests.
- Remove unused imports, an unused AutoGen helper, and notebook-only development dependencies; use the public paper-task helper in the active experiment runner.
- Align contributor instructions and architecture/examples documentation with the own-agent workflow; correct API examples and metric units, run lint and format checks on examples, and use a matching calculator task in the benchmark quickstart.
- Set the async fixture loop scope explicitly and ignore generated coverage/JUnit output.

### Added
- **Test your own agent** with `agentprobe.testing.probe()`: wrap the tools your agent already uses (plain sync/async functions and LangChain tools; originals are never modified), declare deterministic fault scenarios, run the agent, and assert on the outcome (#63, #96)
- Scenarios: `fail_first(n)`, `fail_times(n).then_succeed()`, `always_fail()`, `returns(value)`, `slow(ms=...)`, `with_probability(p, seed=...)`, with any named fault or exception. Rules for one tool form a sequence (#65)
- **Scenario evidence**: each scenario records whether its tool was reached and its fault injected. A required scenario that never fired is `not_exercised` and fails the test, so an agent that bypasses a tool can't pass (#94)
- `ProbeResult` separates observed events (call log, answer, exception, timing, evidence), assertion results (`pass`/`fail`/`unassessed`/`error`) and inferred labels (retries, fallbacks); `summary()` and a versioned `to_dict()` (#66)
- Gate on probe exit: passes only when every required scenario was exercised and every required assertion passed
- Instrumentation contract (`docs/instrumentation.md`): no in-place patching, per-probe state, cleanup on exception and cancellation, and late tool calls from a still-running agent recorded as `after_probe_end` and never counted (#95)
- `await p.arun(...)` for async agents and async tests: invocation and any returned awaitable run in one task under one deadline; at the deadline the probe closes first, cancellation is requested with a bounded `cancel_grace`, and `cancel_requested` / `cancelled` (confirmed) / `still_running` are reported separately; cancelling the caller cancels the agent task and re-raises. `p.run()` rejects async agents inside a running loop before creating a coroutine
- Tool arguments are frozen as JSON-safe snapshots at the call boundary (`args`/`kwargs`); live references are kept separately in `raw_args`/`raw_kwargs` and never serialized
- Calls are recorded when they start (`status="pending"`/`"completed"`, entry-allocated `index`, `started_seq`/`completed_seq`); results are frozen snapshots; calls finishing after the probe closed are marked `completed_after_close`
- Answer assertions require an actual answer: a crash, `None`, a still-running or cancelled agent never passes `assert_answer_not_contains`
- Scenario validation at declaration: positive whole-number counts, finite delays and probabilities, and no rules after one that applies to every call
- `to_dict()` is always JSON-safe (schema 1.2)
- Tutorial and example (`examples/your_agent/`): an order-status agent, a deliberately broken version the tests catch, a cache-skipping agent the gate rejects, and a copyable CI workflow (#67)
- `python -m agentprobe.demo --json` writes the demo report as one JSON object (#100, thanks @Zlongyang for the first community contribution)
- `py.typed` marker: type checkers now use AgentProbe's annotations

### Changed
- The GitHub repository is now `chhayankjain/agentprobe-toolkit`, matching the PyPI name; old URLs redirect. The import name stays `agentprobe`
- The README leads with the developer workflow; the benchmark toolkit and paper reproduction moved to a "For researchers" section
- New pull requests automatically request a maintainer review (CODEOWNERS)
- Documented that unfaulted calls still run real tools (with their side effects) and that 1.4 reports are not redacted (#102)


## [1.3.0] - 2026-11-17

Hosted LLM backends, a no-model demo, and fixes from four rounds of code review that
make new benchmark results safer to trust. The published paper batches are unchanged
and still load and resume.

### Added
- `python -m agentprobe.demo`: a 60-second, network-free demo built on `agentprobe.fake_llm.ScriptedToolCallingModel`, a deterministic tool-calling chat model usable in CI and in users' own tests; `examples/quickstart.py` is the copyable version
- `assets/social-preview.png` for the GitHub repository card
- `agentprobe.llm`: provider-agnostic LLM construction (`LLMSpec`, `build_chat_model`, `autogen_llm_config`) with presets for Ollama, OpenAI, Groq, OpenRouter, Cerebras, Mistral, GitHub Models, Google AI Studio (`gemini` = OpenAI-compatible endpoint for AutoGen; `google` = native `langchain-google-genai` for LangGraph/LangChain, new `[google]` extra), Anthropic, and any OpenAI-compatible server.
- `agentprobe.agents.base.message_text()`: adapters return plain text even when a provider (Gemini, Anthropic) returns content blocks
- AutoGen compatibility shim restores provider extras (Gemini 3 thought signatures) on echoed function calls, which Gemini otherwise rejects with HTTP 400; Ollama requests are unchanged
- `AutoGenToolAgent(stop_on_final_answer=True)` (set automatically for `--provider gemini` via `autogen_agent_kwargs`) ends the two-agent chat at the first text-only reply, because Gemini rejects requests that end with a model turn; the paper's loop-until-`max_turns` behaviour remains the default
- All five experiment scripts take `--provider/--model/--base-url/--api-key-env/--rpm/--max-retries/--llm-timeout`; the default is byte-for-byte the paper's Ollama setup
- LLM backend errors (429/5xx/connection) are distinguished from injected tool faults: the runner re-dispatches the trial (`BenchmarkConfig.backend_error_retries`, backoff), then records `status="backend_error"` / category `infrastructure`; `MetricsCalculator` excludes them and reports `n_backend_errors`. `FailureClassifier` gains `FailureCategory.INFRASTRUCTURE`
- `experiment_metadata.json` records the exact backend (`llm_provider`, `model_id`, `llm_base_url`, key variable name, throttle); `--resume` rejects a changed provider or endpoint

### Changed
- `run_benchmarks.py --llm {ollama,openai,anthropic}` replaced by `--provider` (same defaults)
- Provider exceptions (`RateLimitError`, `APITimeoutError`, `APIConnectionError`, …) now classify as `infrastructure` instead of `tool_call`
- Behaviour changes to check when upgrading: `run_benchmarks.py` now exits nonzero when any condition is incomplete; `tool_call_count` in Phase 2 records is per trial (condition totals stay in `injector_calls`/`injector_injections`); adapters always install a pass-through injector, so baseline conditions report tool calls; Phase 2 metadata gains `protocol`, and `--resume` rejects a changed protocol

### Fixed
- The legacy benchmark CLI stops on incomplete conditions and exits nonzero after saving partial results. Its table shows executed/planned and excluded counts; empty samples display N/A. Recovery summaries use the assessable denominator, and v2 records preserve backend retry counts.
- LLM backend errors are identified at the **LLM-call boundary**, not by exception class: adapters wrap every model call in `agentprobe.llm.llm_call_boundary()` (LangGraph node, LangChain middleware, AutoGen shim). A real tool failing with `httpx.ReadTimeout` is now measured as an agent failure instead of being excluded. Custom agents should wrap their model calls the same way
- AutoGen no longer silently executes only the first of several parallel tool calls: such a turn raises `ParallelToolCallsUnsupported` and the trial is recorded as `status="harness_error"` (excluded, `n_harness_errors`). OpenAI, Groq and OpenRouter are asked for `parallel_tool_calls=false`
- Per-trial tool evidence: `trial_tool_calls`, `trial_injections`, `recovery_tool_calls`, `recovery_injections` on every record, from injector snapshots around each invocation; `tool_call_count` is now per trial. `injector_calls`/`injector_injections` keep their condition-total meaning. Adapters always install a pass-through injector, so baseline conditions also record tool calls (previously 0)
- `cohens_d` returns `(nan, "undefined")` for different constant groups instead of a misleading 0; statistics reject non-finite inputs; Kruskal-Wallis on all-identical data returns p=1 instead of raising
- Malformed payloads come from a separate seeded stream, reproducible from the seed without shifting the injection schedule
- `drop_fraction=0` keeps every message (any positive fraction still drops at least one)
- The source distribution includes the experiment scripts and examples its tests import; CI runs the test suite from the extracted sdist
- Experiment metadata records a `protocol` label (`phase1-discovery` with unmatched probabilities vs `phase2-matched`); `run_benchmarks.py` prints which one it runs, and resume rejects a protocol change
- AutoGen conditions no longer construct an unused LangChain model, so hosted AutoGen runs work in the `openai<1` environment
- LLM backend errors during the *recovery* call are retried with backoff and, if persistent, recorded as `recovery_status="unassessed"`; `MetricsCalculator` drops those from the recovery-rate denominator and reports `n_recovery_unassessed`. `RunResult` gains `recovery_status` (`recovered`/`failed`/`unassessed`/`not_attempted`/`n/a`)
- `run_experiment_v2.py --resume` with a subset of conditions rebuilds `combined_summary.json` and `manifest.json` from every checkpointed condition on disk instead of shrinking them; the manifest lists `completed_conditions` and `n_records_this_invocation`
- AutoGen honours `temperature`, `--rpm` (shim-level throttle) and `--max-retries` (bounded by the compatibility shim); `experiment_metadata.json` records `llm_autogen_effective`
- AutoGen retries are bounded for every transient provider error. pyautogen 0.1.14 retries `ServiceUnavailableError`, `APIConnectionError` and `APIError` with no limit; the shim now counts consecutive failures and raises `BackendRetriesExhausted` (an infrastructure error) after `--max-retries`
- AutoGen thought-signature cache keeps every repeated identical call in order and is cleared per conversation, so restored history can no longer swap ids/signatures
- `AgentProbeTracer` reuses one process-wide provider (or an SDK provider the application already registered); later instances attach exporters to it instead of creating a provider library spans bypass, and `shutdown()` only flushes
- `--resume` also rejects changed timeout, retry, throttle and temperature settings, comparing by key presence so an explicit `None` (unlimited RPM) counts as a value. Batches recorded before provider support are checked against the documented defaults they ran with (Ollama, temperature 0, no throttle)
- AutoGen throttle, retry limit and tool-call cache are per agent and per thread: each agent activates its own settings at the start of every conversation, `None` clears a previous throttle, the pyautogen retry window is set in each agent's own `llm_config` instead of on the global class, and concurrent conversations in different threads no longer share state
- Demo uses a calculator prompt matching its scripted model and reports correct final answers separately from execution success

## [1.2.0] - 2026-11-14

First release published to PyPI, as **`agentprobe-toolkit`** (the name `agentprobe` was already
taken by an unrelated project). The import name is unchanged: `import agentprobe`.

### Added
- PyPI distribution `agentprobe-toolkit` with Trusted Publishing release workflow
  (`.github/workflows/release.yml`): tag `vX.Y.Z` publishes to PyPI; manual dispatch publishes to TestPyPI
- Packaging check in CI (`python -m build`, `twine check`, wheel import smoke test)
- Python 3.14 in the CI matrix and classifiers

### Fixed
- `from agentprobe import ToolFailureInjector, AgentProbeTracer` (the README example) now works: the top-level package re-exports the public API
- Removed the duplicate Pydantic `ToolFailureConfig`/`ToolFailureType` in `agentprobe/injector/types.py`; `agentprobe.injector` now exports the dataclass versions the injector actually accepts (the old export crashed `ToolFailureInjector` with `AttributeError`)
- `BaseAgent` protocol now matches the real adapter surface (`invoke(dict) -> dict`, `reset()`, `get_injector_summary()`); all three adapters gained `reset()` and an `injector` property
- Dependency floors corrected: `langchain>=1.0.0`, `langchain-core>=1.0.0`, `langgraph>=1.0.0` (the LangChain adapter needs `create_agent`); unused `langchain-community` dropped
- Tracer no longer suggests a non-existent `agentprobe[otlp]` extra
- Runner progress summary uses the same ceiling nearest-rank P99 as `MetricsCalculator`

### Added (API)
- `agentprobe[autogen]` extra pinning `pyautogen==0.1.14` and `openai<1.0`
- `agentprobe.benchmark.tasks.get_task()` and `corpus_task_fn()` to run the 55-query graded corpora through `BenchmarkRunner`; `run_benchmarks.py --query-set corpus`
- `runner.PAPER_PROMPTS` / `paper_task_fn` naming the five-prompt set every published batch used

### Removed
- Dead Pydantic schema layer `agentprobe/benchmark/types.py` (`TaskType`, `AgentFramework`, Pydantic `RunResult`/`BenchmarkConfig`/`BenchmarkResults`/`BenchmarkMetrics`) that no runner code used

### Changed
- README rewritten to state which taxonomy categories, frameworks, and LLM backends were actually validated in the paper, with metric definitions
- `ROADMAP.md` replaced by a pointer to the GitHub roadmap issue (the file described a pre-v1.0 build plan)

## [1.1.0] - 2026-11-14

### Added
- Phase 2 experiment runner (`run_experiment_v2.py`) with matched injection probability and answer-presence grading
- Qualification runner (`run_qualification.py`) for deterministic pipeline validation at p=1.0
- Model smoke test (`smoke_test_model.py`) for quick tool-calling verification
- AutoGen diagnostic (`autogen_diagnostic.py`) comparing direct OpenAI client vs AutoGen shim
- Cross-model analysis tooling (`analyze_cross_model.py`) and data-driven figure generation
- Checkpoint validation with exact record count, run ID range, and condition identity matching
- Resume safety: reject metadata mismatches (model, seed, n_runs) on experiment resume
- Full experiment data for Llama 3.1, Mistral 7B, and Qwen2.5 7B (Phase 2, qualification, diagnostic)
- Phase 1 data archived to `experiments/archive_v1/`
- Paper source map (`paper_source_map.json`) for batch-to-table provenance
- ORCID identifiers in CITATION.cff

### Fixed
- AutoGen adapter: patch legacy `functions`/`function_call` format for Ollama compatibility
- AutoGen output extraction: select correct assistant message instead of empty tool-call messages
- Numeric answer validation: word-boundary regex to prevent substring false positives (e.g., "12000" matching "120000")
- `recovery_rate` returns `None` (not `0.0`) when no failures occur in a condition
- P99 percentile: switched from floor-based to ceiling nearest-rank
- `None` recovery rate handling in reporting scripts, plot generation, and analysis
- Figure output directory aligned with LaTeX `\graphicspath`
- Pilot hung-invocation control flow and empty-output parsing

### Changed
- Manuscript updated with two-phase study, cross-model validation, and MDPI/MAKE formatting
- Phase 1 latency ratio corrected from 7x to 11x
- Scope claims narrowed to experimentally validated fault categories
- Removed uninformative chi-square test from identical-implementation comparison
- LaTeX commands use `\providecommand` for Overleaf/mdpi.cls compatibility

## [1.0.0] - 2026-09-13

### Added
- Initial public release
- Five-category failure taxonomy (Tool Call, Orchestration, Context, Latency, Consistency)
- Tool call fault injectors: timeout, malformed output, rate limit, API error
- OpenTelemetry-based tracing with Jaeger/Prometheus integration
- Benchmark runner with MTTR, failure rate, and P99 latency metrics
- Agent adapters for LangGraph, LangChain, and AutoGen
- 15 benchmark queries across tool use, RAG, and multi-agent tasks
- Phase 1 experiment data (1,050 runs)
