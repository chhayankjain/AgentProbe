"""Failure injection modules for AgentProbe.

The tool-call injector is the one used in the paper's experiments. The
orchestration, context, and latency injectors are implemented and unit-tested
but have not yet been used in a published experiment.

Usage::

    from agentprobe.injector import ToolFailureConfig, ToolFailureInjector, ToolFailureType

    injector = ToolFailureInjector(
        ToolFailureConfig(failure_type=ToolFailureType.TIMEOUT, failure_probability=0.2, seed=42)
    )
"""

from agentprobe.injector.context_failure import (
    ContextFailureConfig,
    ContextFailureInjector,
    ContextFailureType,
)
from agentprobe.injector.latency_failure import (
    LatencyFailureConfig,
    LatencyFailureInjector,
    LatencyFailureType,
)
from agentprobe.injector.orchestration_failure import (
    OrchestrationFailureConfig,
    OrchestrationFailureInjector,
    OrchestrationFailureType,
)
from agentprobe.injector.tool_failure import (
    InjectionResult,
    ToolCallAPIError,
    ToolCallError,
    ToolCallMalformedOutput,
    ToolCallRateLimited,
    ToolCallTimeout,
    ToolFailureConfig,
    ToolFailureInjector,
    ToolFailureType,
)

__all__ = [
    # Tool-call faults (Category 1) — used in the paper
    "ToolFailureConfig",
    "ToolFailureInjector",
    "ToolFailureType",
    "InjectionResult",
    "ToolCallError",
    "ToolCallTimeout",
    "ToolCallMalformedOutput",
    "ToolCallRateLimited",
    "ToolCallAPIError",
    # Orchestration faults (Category 2) — implemented, not yet experimentally validated
    "OrchestrationFailureConfig",
    "OrchestrationFailureInjector",
    "OrchestrationFailureType",
    # Context faults (Category 3) — implemented, not yet experimentally validated
    "ContextFailureConfig",
    "ContextFailureInjector",
    "ContextFailureType",
    # Latency faults (Category 4) — implemented, not used in the experiments
    "LatencyFailureConfig",
    "LatencyFailureInjector",
    "LatencyFailureType",
]
