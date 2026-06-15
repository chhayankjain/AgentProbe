"""AgentProbe-facing adapters: BaseAgent implementations for each framework."""

from agentprobe.agents.base import AgentResult
from .result_mapper import FailureHook, NoOpFailureHook, to_agent_result

__all__ = [
    "AgentResult",
    "FailureHook",
    "NoOpFailureHook",
    "to_agent_result",
]
