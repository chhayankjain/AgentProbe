"""Mapping from PipelineRunResult to AgentResult + failure-injection seam."""

import time
from typing import Any, Callable, Protocol

from pydantic import BaseModel, Field

from core.schemas import PipelineRunResult


class FailureHook(Protocol):
    """Duck-typed protocol for failure injection (satisfied by AgentProbe's ToolFailureInjector).

    Defaults to a no-op; AgentProbe threads its injector through this seam during benchmarking.
    """
    def maybe_fail(self, tool_name: str) -> None:
        """Optionally inject a failure (e.g., timeout, malformed output) for the named tool."""
        ...


class NoOpFailureHook:
    """Default failure hook: no-op."""
    def maybe_fail(self, tool_name: str) -> None:
        pass


class AgentResult(BaseModel):
    """AgentProbe's canonical result type (imported here for type clarity)."""
    success: bool
    output: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)
    retries: int = Field(default=0, ge=0)
    error_message: str | None = None


def to_agent_result(
    run_result: PipelineRunResult,
    duration_ms: float,
    retries: int,
) -> AgentResult:
    """Map PipelineRunResult -> AgentResult.

    Args:
        run_result: Output from a pipeline.run() call.
        duration_ms: Wall-clock runtime in milliseconds.
        retries: Number of retry/retrain attempts (typically run_result.attempts - 1).

    Returns:
        AgentResult conforming to AgentProbe's contract.
    """
    return AgentResult(
        success=run_result.success,
        output=run_result.report_markdown,
        duration_ms=duration_ms,
        retries=retries,
        error_message=run_result.error,
    )
