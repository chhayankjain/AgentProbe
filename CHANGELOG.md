# Changelog

## [Unreleased]

### Fixed
- `from agentprobe import ToolFailureInjector, AgentProbeTracer` (the README example) now works: the top-level package re-exports the public API
- Removed the duplicate Pydantic `ToolFailureConfig`/`ToolFailureType` in `agentprobe/injector/types.py`; `agentprobe.injector` now exports the dataclass versions the injector actually accepts (the old export crashed `ToolFailureInjector` with `AttributeError`)
- `BaseAgent` protocol now matches the real adapter surface (`invoke(dict) -> dict`, `reset()`, `get_injector_summary()`); all three adapters gained `reset()` and an `injector` property
- Dependency floors corrected: `langchain>=1.0.0`, `langchain-core>=1.0.0`, `langgraph>=1.0.0` (the LangChain adapter needs `create_agent`); unused `langchain-community` dropped
- Tracer no longer suggests a non-existent `agentprobe[otlp]` extra
- Runner progress summary uses the same ceiling nearest-rank P99 as `MetricsCalculator`

### Added
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
