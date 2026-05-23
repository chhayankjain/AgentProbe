"""Abstract base protocol and shared result type for all AgentProbe agent adapters.

Usage::

    from agentprobe.agents.base import BaseAgent, AgentResult

    class MyAgent:
        def run(self, task_input: str) -> AgentResult:
            ...
        def reset(self) -> None:
            ...
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """Outcome of a single ``BaseAgent.run()`` call.

    Args:
        success: Whether the agent completed the task without unrecovered failure.
        output: Final text output from the agent. ``None`` on unrecovered failure.
        duration_ms: Wall-clock time for the run in milliseconds.
        retries: Number of retry attempts made during the run.
        error_message: Human-readable error detail when ``success`` is ``False``.
    """

    success: bool
    output: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)
    retries: int = Field(default=0, ge=0)
    error_message: str | None = None


@runtime_checkable
class BaseAgent(Protocol):
    """Structural protocol satisfied by all AgentProbe agent adapters.

    Concrete adapters (LangGraph, LangChain, AutoGen) must implement both
    methods. The protocol is ``runtime_checkable`` so ``isinstance`` checks
    work in the benchmark runner without importing concrete adapter classes.
    """

    def run(self, task_input: str) -> AgentResult:
        """Execute the task described by *task_input* and return the outcome."""
        ...

    def reset(self) -> None:
        """Clear any internal state so the next ``run()`` starts fresh."""
        ...
