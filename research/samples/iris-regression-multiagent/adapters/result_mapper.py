"""Mapping from PipelineRunResult to AgentResult + failure-injection seam."""

from __future__ import annotations

from typing import Protocol

from agentprobe.agents.base import AgentResult
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
