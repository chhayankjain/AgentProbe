"""Agent implementations for AgentProbe benchmarking (the ``research`` extra)."""

from agentprobe._extras import missing_extra

try:
    from agentprobe.agents.base import AgentResult, BaseAgent
except ModuleNotFoundError as exc:  # an optional dependency is not installed
    raise missing_extra(exc, "agentprobe.agents", "research") from exc

__all__ = ["AgentResult", "BaseAgent"]
