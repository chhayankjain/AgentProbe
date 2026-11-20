"""AgentProbe quickstart: inject faults into an agent and measure it, with no model needed.

    pip install "agentprobe-toolkit[research]"
    python examples/quickstart.py

This uses the same building blocks as ``python -m agentprobe.demo``, spelled out
step by step so you can copy the pieces into your own code. Replace the scripted model
with a real one (``agentprobe.llm.build_chat_model``) and the rest is unchanged.
"""

from __future__ import annotations

from agentprobe import (
    AgentProbeTracer,
    BenchmarkConfig,
    BenchmarkRunner,
    MetricsCalculator,
    ToolFailureConfig,
    ToolFailureType,
)
from agentprobe.agents.langgraph_agent import LangGraphToolAgent
from agentprobe.fake_llm import ScriptedToolCallingModel

# 1. Tracing (console export off; set otlp_endpoint="http://localhost:4317" for Jaeger).
tracer = AgentProbeTracer(service_name="quickstart", console=False)

# 2. Decide what to break. Seeded, so the same calls fault on every run.
faults = ToolFailureConfig(
    failure_type=ToolFailureType.RATE_LIMIT,  # or TIMEOUT, MALFORMED_OUTPUT, API_ERROR
    failure_probability=0.3,  # 30% of tool calls
    seed=42,
)

# 3. Build the agent. Replace the scripted model with build_chat_model(LLMSpec(...))
#    to use a real backend.
agent = LangGraphToolAgent(llm=ScriptedToolCallingModel(), failure_config=faults)


def calculator_task(run_id: int, _task: str) -> dict[str, str | int]:
    """Use the task the scripted model's fixed calculator expression answers."""
    return {"input": "Calculate 15% of 847.", "run_id": run_id}


# 4. Run a small benchmark: 10 matching calculator tasks, one retry after a failure.
results = BenchmarkRunner(
    agent=agent,
    config=BenchmarkConfig(n_runs=10, failure_config=faults, recovery_attempts=1),
    task_fn=calculator_task,
).run()

# 5. Metrics and the per-call injection log, each entry linked to an OpenTelemetry span.
m = MetricsCalculator().compute(results)
print(
    f"\nfailure rate {m.failure_rate:.0%} | recovery rate {m.recovery_rate} | "
    f"P99 {m.p99_latency_ms:.0f} ms"
)
for rec in agent.injector.injection_log:
    print(
        f"  call {rec.call_index}: {rec.tool_name:<10} "
        f"injected={rec.injected!s:<5} span={rec.span_id}"
    )

tracer.shutdown()
