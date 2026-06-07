"""AgentProbe-facing adapters: BaseAgent implementations for each framework."""

from .result_mapper import AgentResult, FailureHook, NoOpFailureHook, to_agent_result

__all__ = [
    "AgentResult",
    "FailureHook",
    "NoOpFailureHook",
    "to_agent_result",
]
