"""Pydantic models and enums for the AgentProbe failure-injection subsystem.

Usage::

    from agentprobe.injector.types import ToolFailureConfig, ToolFailureType

    cfg = ToolFailureConfig(
        failure_type=ToolFailureType.TIMEOUT,
        failure_probability=0.2,
        timeout_seconds=5.0,
        seed=42,
    )
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ToolFailureType(str, Enum):
    """The kind of failure to inject into a tool call."""

    NONE = "none"
    TIMEOUT = "timeout"
    MALFORMED_OUTPUT = "malformed_output"
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    SERVER_ERROR = "server_error"


class ToolFailureConfig(BaseModel):
    """Configuration for :class:`~agentprobe.injector.tool_failure.ToolFailureInjector`.

    Args:
        failure_type: Which failure mode to simulate.
        failure_probability: Fraction of tool calls that will be intercepted (0.0–1.0).
        timeout_seconds: Simulated timeout duration when ``failure_type`` is TIMEOUT.
        seed: Random seed for reproducible injection sequences. ``None`` means unseeded.
        max_retries: How many retry attempts an agent is allowed before the run is
            considered failed.
    """

    failure_type: ToolFailureType = ToolFailureType.NONE
    failure_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    seed: int | None = None
    max_retries: int = Field(default=3, ge=0)
