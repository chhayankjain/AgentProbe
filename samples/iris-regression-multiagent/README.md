# iris-regression-multiagent

A **multi-framework agentic system** for Iris linear regression, designed as a test target for [AgentProbe](https://github.com/anthropics/agentprobe) — a reliability and observability benchmarking toolkit for production LLM agents.

## Project Purpose

This repo demonstrates how to solve a single, deterministic ML problem (**predict petal width from Iris features**) using three different multi-agent orchestration frameworks:

- **LangGraph** (graph-based state machine)
- **AutoGen/ag2** (conversational multi-agent with group chat)
- **CrewAI** (role/task-based crew framework)

The goal is to provide a rich, diverse agentic surface for AgentProbe to probe — testing how each framework's orchestration patterns (tool calls, handoffs, loops, context) respond to injected failures (timeouts, malformed outputs, rate limits, dropped handoffs, latency spikes).

## Repository Structure

```
core/                  # Pure Python ML pipeline (no LLM/agent imports, fully deterministic)
tools/                 # Shared tool contracts (framework-agnostic)
agents_langgraph/      # LangGraph implementation [NOT YET IMPLEMENTED]
agents_autogen/        # AutoGen/ag2 implementation [NOT YET IMPLEMENTED]
agents_crewai/         # CrewAI implementation [NOT YET IMPLEMENTED]
adapters/              # AgentProbe BaseAgent adapters
examples/              # Demo and reference scripts [NOT YET IMPLEMENTED]
tests/                 # Comprehensive test suite
```

## Quick Start

### Installation

```bash
# Install core + development dependencies
pip install -e ".[dev]"

# Or install a specific framework slice
pip install -e ".[langgraph]"   # for LangGraph
pip install -e ".[autogen]"     # for AutoGen/ag2
pip install -e ".[crewai]"      # for CrewAI
pip install -e ".[all]"         # all frameworks + deps
```

### Running Core Tests (no LLM)

The deterministic core is fully tested and requires no API keys:

```bash
pytest tests/ -v
```

This runs 38 tests covering:
- Data loading and seeded reproducibility
- Linear regression fitting
- Evaluation metrics (R², MAE, RMSE)
- Report generation
- End-to-end oracle pipeline

### Running the Reference Pipeline

```python
from core.pipeline import run_pipeline
result = run_pipeline()
print(f"Success: {result.success}")
print(f"Final R²: {result.final_r2:.4f}")
print(f"Attempts: {result.attempts}")
print(result.report_markdown)
```

Expected output:
- **Success**: `True` (assumes default config)
- **Final R²**: `≈ 0.92` (3-feature model on test set)
- **Attempts**: `2` (first escalation rejected, second accepted)

## The ML Task

**Target**: `petal width (cm)` (continuous)  
**Features**: `sepal length (cm)`, `sepal width (cm)`, `petal length (cm)`  
**Algorithm**: Linear Regression (sklearn)  
**R² Threshold**: 0.85 (configurable)

Feature escalation strategy (built-in retrain loop):
1. **Attempt 1**: Single low-correlation feature → R² ≈ 0.63 → **REJECTED**
2. **Attempt 2**: All 3 features → R² ≈ 0.93 → **ACCEPTED** ✓

This natural, deterministic loop exercises agentic orchestration patterns (conditional routing, state transitions, inter-agent handoffs) that AgentProbe will probe.

## Shared Tool Contracts

All four tools are defined once in `tools/specs.py` and mechanically translated into each framework's tool format:

| Tool | Input | Output |
|---|---|---|
| `load_iris_data` | Feature subset, split ratio | Dataset ID (cached) |
| `fit_linear_regression` | Dataset ID | Model ID (cached) + coefficients |
| `evaluate_regression` | Model ID, dataset ID | R², MAE, RMSE, verdict |
| `write_report` | Model ID, eval output, attempt | Markdown report |

**Why this design?**
- Identical validation, error surfaces, and numerics across all three frameworks
- Only orchestration shell differs (the actual subject of AgentProbe testing)
- Easy to inject failures at tool boundaries

## Agent Decomposition (shared across all 3 frameworks)

Four thin agents, each with one primary responsibility:

| Agent | Role | Tool(s) | AgentProbe injection point |
|---|---|---|---|
| **DataPrepAgent** | Feature selection, dataset loading | `load_iris_data` | Tool timeout, malformed output, handoff drops |
| **ModelTrainerAgent** | Model fitting | `fit_linear_regression` | Tool rate-limit, malformed coefficients, handoff drops |
| **EvaluatorAgent** | Quality assessment, retrain decision | `evaluate_regression` | API errors, loop bounds, handoff drops |
| **ReportWriterAgent** | Final documentation | `write_report` | Server errors, context overflow |

**Philosophy**: Agents orchestrate; tools compute. Business logic lives in `core/`, not in prompts.

## Development Status

### Completed (Slice 1)
✅ Core ML pipeline (`core/` + `tools/`)  
✅ Shared tool contracts and schemas  
✅ 38 passing deterministic tests  
✅ Adapters stub (result mapper)  

### Planned (Slices 2-6)
🔲 LangGraph agent implementation  
🔲 AutoGen/ag2 agent implementation  
🔲 CrewAI agent implementation  
🔲 Smoke tests with mocked LLMs  
🔲 Integration examples with AgentProbe  

## Building Next: LangGraph Agent (Slice 2)

The LangGraph implementation (`agents_langgraph/`) will:
1. Define a `StateGraph` with `IrisGraphState` (TypedDict)
2. Implement node functions: `data_prep_node`, `train_node`, `evaluate_node`, `report_node`
3. Wire conditional edge: `evaluate → {retrain, accept, abort}` (the orchestration-loop surface)
4. Wrap tools as `StructuredTool` instances
5. Return `PipelineRunResult` from `pipeline.run(task_input: str)`
6. Adapt to AgentProbe's `BaseAgent` protocol via `adapters/langgraph_adapter.py`

Then CrewAI and AutoGen follow the same pattern, with framework-specific orchestration idioms.

## Configuration

Set LLM backend via environment variables (when agent slices are built):

```bash
# Ollama (local, zero cost — recommended)
export OLLAMA_BASE_URL="http://localhost:11434"

# Or OpenAI
export OPENAI_API_KEY="sk-..."

# Or Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

Default config uses Ollama + Llama-3 for zero-cost, reproducible benchmarking.

## Testing with AgentProbe

When complete, this repo integrates with AgentProbe:

```python
from adapters import LangGraphIrisAgent
from agentprobe.benchmark import BenchmarkRunner, BenchmarkConfig, ToolFailureConfig
from agentprobe.injector.types import ToolFailureType

# Create agent
agent = LangGraphIrisAgent(llm=...)

# Run benchmark with injected failures
config = BenchmarkConfig(
    framework="langgraph",
    task="iris_regression",
    n_runs=50,
    failure_config=ToolFailureConfig(
        failure_type=ToolFailureType.TIMEOUT,
        failure_probability=0.3,
    ),
)

runner = BenchmarkRunner()
results = runner.run(agent, config)
metrics = runner.compute_metrics(results)

print(f"Failure rate under timeout injection: {metrics.failure_rate:.2%}")
print(f"MTTR: {metrics.mttr_ms:.1f}ms")
print(f"P99 latency: {metrics.p99_latency_ms:.1f}ms")
```

## License

MIT

## Related

- [AgentProbe](https://github.com/anthropics/agentprobe) — Benchmarking toolkit this repo targets
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/) — Graph-based agent orchestration
- [AutoGen/ag2 Docs](https://docs.ag2.ai/) — Conversational multi-agent framework
- [CrewAI Docs](https://docs.crewai.com/) — Role/task-based agent crews
