"""Observability modules for AgentProbe."""

from agentprobe._extras import missing_extra

try:
    from agentprobe.observer.tracer import AgentProbeTracer, get_tracer
except ModuleNotFoundError as exc:  # an optional dependency is not installed
    raise missing_extra(exc, "agentprobe.observer") from exc

__all__ = ["AgentProbeTracer", "get_tracer"]
