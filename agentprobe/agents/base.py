"""Shared protocol and result helper for all AgentProbe agent adapters.

Every adapter exposes the same minimal surface so that
:class:`~agentprobe.benchmark.runner.BenchmarkRunner` can drive any framework
without knowing which one it is:

* ``framework`` — a short string identifier (``"langgraph"``, ``"langchain"``, ``"autogen"``).
* ``invoke(input)`` — run one task. ``input`` is a dict with at least an
  ``"input"`` key (the user query) and optionally ``"context"`` (RAG context).
  Returns a dict with at least ``"output"``, ``"latency_ms"``, and ``"framework"``.
* ``reset()`` — clear per-conversation state so the next ``invoke()`` starts fresh.
  This does **not** reset the fault injector: injection state is condition-wide
  by design (see :meth:`ToolFailureInjector.reset`).
* ``get_injector_summary()`` — aggregate injection statistics for the condition.

Usage::

    from agentprobe.agents import BaseAgent

    class MyAgent:
        framework = "my_framework"

        def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
            ...

        def reset(self) -> None:
            ...

        def get_injector_summary(self) -> dict[str, Any]:
            return {}

    assert isinstance(MyAgent(), BaseAgent)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """Optional structured view of one ``invoke()`` outcome.

    Adapters return plain dicts from ``invoke()`` so that framework-specific
    extras (e.g. AutoGen's ``tool_calls`` count) pass through untouched.
    ``AgentResult`` is a typed convenience for code that wants validation,
    for example external adapters that wrap a non-LLM pipeline.

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

    @classmethod
    def from_invoke(cls, result: dict[str, Any]) -> AgentResult:
        """Build an ``AgentResult`` from an adapter's ``invoke()`` return dict."""
        output = result.get("output")
        return cls(
            success=bool(output),
            output=str(output) if output else None,
            duration_ms=float(result.get("latency_ms", 0.0)),
            retries=int(result.get("retries", 0)),
            error_message=result.get("error_message"),
        )


@runtime_checkable
class BaseAgent(Protocol):
    """Structural protocol satisfied by all AgentProbe agent adapters.

    The protocol is ``runtime_checkable`` so ``isinstance`` checks work in the
    benchmark runner without importing concrete adapter classes.
    """

    framework: str

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        """Run one task and return a dict with ``output``, ``latency_ms``, ``framework``."""
        ...

    def reset(self) -> None:
        """Clear per-conversation state so the next ``invoke()`` starts fresh."""
        ...

    def get_injector_summary(self) -> dict[str, Any]:
        """Return aggregate fault-injection statistics (empty if no injector)."""
        ...
