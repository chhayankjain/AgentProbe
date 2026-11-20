"""The exceptions AgentProbe injects into tool calls.

Dependency-free, so ``agentprobe.testing`` works on a minimal install. They are
also importable from :mod:`agentprobe.injector.tool_failure`, where they used to
live; both paths give the same classes.
"""

from __future__ import annotations

__all__ = [
    "ToolCallError",
    "ToolCallTimeout",
    "ToolCallMalformedOutput",
    "ToolCallRateLimited",
    "ToolCallAPIError",
]


class ToolCallError(Exception):
    """Base class for injected tool call errors."""


class ToolCallTimeout(ToolCallError, TimeoutError):
    """Raised when a tool call exceeds its timeout deadline."""


class ToolCallMalformedOutput(ToolCallError):
    """Raised when a tool returns unparseable or corrupt output."""


class ToolCallRateLimited(ToolCallError):
    """Raised to simulate HTTP 429 rate-limit responses."""

    def __init__(self, retry_after: float = 60.0) -> None:
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry-After: {retry_after}s (HTTP 429)")


class ToolCallAPIError(ToolCallError):
    """Raised to simulate HTTP 5xx server-side API errors."""

    def __init__(self, status_code: int = 500) -> None:
        self.status_code = status_code
        super().__init__(f"Tool API returned HTTP {status_code}")
