"""Tool call failure injection for agentic LLM systems.

Failure taxonomy — Tool Call Failures (Category 1):
  - TIMEOUT        : Tool call exceeds wall-clock deadline
  - MALFORMED_OUTPUT: Tool returns non-parseable or corrupt output
  - RATE_LIMIT     : Tool API returns HTTP 429 / rate-limit error
  - API_ERROR      : Tool API returns HTTP 5xx / generic server error

Usage — decorator::

    injector = ToolFailureInjector(
        config=ToolFailureConfig(
            failure_type=ToolFailureType.TIMEOUT,
            failure_probability=0.5,
            timeout_seconds=5.0,
        )
    )

    @injector.wrap
    def web_search(query: str) -> str:
        return _real_search(query)

Usage — explicit call::

    result = injector.call(web_search, "climate change")

Usage — LangChain tool wrapping::

    from langchain_core.tools import tool
    from agentprobe.injector.tool_failure import inject_langchain_tool

    @tool
    def web_search(query: str) -> str:
        \"\"\"Search the web.\"\"\"
        return _real_search(query)

    faulty_tool = inject_langchain_tool(web_search, injector)
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

from opentelemetry.trace import Status, StatusCode

from agentprobe.observer.tracer import get_tracer

T = TypeVar("T")

_tracer = get_tracer(__name__)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class ToolCallError(Exception):
    """Base class for injected tool call errors."""


class ToolCallTimeout(ToolCallError):
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


# ---------------------------------------------------------------------------
# Enums and config
# ---------------------------------------------------------------------------


class ToolFailureType(str, Enum):
    NONE = "none"
    TIMEOUT = "timeout"
    MALFORMED_OUTPUT = "malformed_output"
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"


@dataclass
class ToolFailureConfig:
    """Configuration for tool failure injection.

    Attributes:
        failure_type: Which failure mode to inject.
        failure_probability: Probability [0, 1] of injecting on each call.
        timeout_seconds: Simulated wall-clock timeout (TIMEOUT mode).
        malformed_payload: Custom bad payload string (MALFORMED_OUTPUT mode).
            If None, a random malformed string is chosen.
        retry_after_seconds: Retry-After value in seconds (RATE_LIMIT mode).
        error_code: HTTP status code to simulate (API_ERROR mode).
        seed: RNG seed for reproducible experiments.
    """

    failure_type: ToolFailureType = ToolFailureType.NONE
    failure_probability: float = 1.0
    timeout_seconds: float = 30.0
    malformed_payload: str | None = None
    retry_after_seconds: float = 60.0
    error_code: int = 500
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.failure_probability <= 1.0:
            raise ValueError("failure_probability must be in [0, 1]")


# ---------------------------------------------------------------------------
# Injection result
# ---------------------------------------------------------------------------


@dataclass
class InjectionResult:
    """Record of a single tool call invocation under injection."""

    tool_name: str
    failure_type: ToolFailureType
    injected: bool
    exception: Exception | None = None
    latency_ms: float = 0.0
    span_id: str | None = None
    call_index: int = 0


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------


class ToolFailureInjector:
    """Injects tool call failures into LangGraph / LangChain / AutoGen agents.

    Thread-safe for read operations on :attr:`injection_log`; writes are
    append-only and safe for single-threaded benchmark runs.
    """

    def __init__(self, config: ToolFailureConfig | None = None) -> None:
        self.config = config or ToolFailureConfig()
        self._rng = random.Random(self.config.seed)
        self._log: list[InjectionResult] = []
        self._call_count = 0

    # ------------------------------------------------------------------
    # Decorator API
    # ------------------------------------------------------------------

    def wrap(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Wrap a synchronous tool function with failure injection."""

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self._execute_sync(fn, fn.__name__, *args, **kwargs)  # type: ignore[return-value]

        return wrapper  # type: ignore[return-value]

    def wrap_async(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Wrap an async tool function with failure injection."""

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await self._execute_async(fn, fn.__name__, *args, **kwargs)  # type: ignore[return-value]

        return wrapper  # type: ignore[return-value]

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Directly invoke a function with failure injection (no decoration needed)."""
        return self._execute_sync(fn, fn.__name__, *args, **kwargs)  # type: ignore[return-value]

    async def call_async(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Directly invoke an async function with failure injection."""
        return await self._execute_async(fn, fn.__name__, *args, **kwargs)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # LangChain integration
    # ------------------------------------------------------------------

    def inject_langchain_tool(self, tool: Any) -> Any:
        """Wrap a LangChain BaseTool's _run/_arun with failure injection.

        Returns the same tool object with monkey-patched run methods.
        This preserves the tool's schema and metadata for the agent.
        """
        original_run = tool._run
        original_arun = tool._arun

        @functools.wraps(original_run)
        def patched_run(*args: Any, **kwargs: Any) -> Any:
            return self._execute_sync(original_run, tool.name, *args, **kwargs)

        @functools.wraps(original_arun)
        async def patched_arun(*args: Any, **kwargs: Any) -> Any:
            return await self._execute_async(original_arun, tool.name, *args, **kwargs)

        tool._run = patched_run
        tool._arun = patched_arun
        return tool

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    @property
    def injection_log(self) -> list[InjectionResult]:
        return list(self._log)

    def reset(self) -> None:
        self._log.clear()
        self._call_count = 0
        self._rng = random.Random(self.config.seed)

    def summary(self) -> dict[str, Any]:
        """Aggregate statistics over all recorded calls."""
        total = len(self._log)
        injected = [r for r in self._log if r.injected]
        latencies = [r.latency_ms for r in self._log]
        by_type: dict[str, int] = {}
        for r in injected:
            key = r.failure_type.value
            by_type[key] = by_type.get(key, 0) + 1

        return {
            "total_calls": total,
            "injected_failures": len(injected),
            "injection_rate": len(injected) / total if total > 0 else 0.0,
            "failures_by_type": by_type,
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "max_latency_ms": max(latencies) if latencies else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _should_inject(self) -> bool:
        if self.config.failure_type == ToolFailureType.NONE:
            return False
        return self._rng.random() < self.config.failure_probability

    def _execute_sync(self, fn: Callable, name: str, *args: Any, **kwargs: Any) -> Any:
        self._call_count += 1
        idx = self._call_count
        inject = self._should_inject()
        start = time.perf_counter()

        with _tracer.start_as_current_span(
            f"tool.call.{name}",
            attributes={
                "tool.name": name,
                "tool.call_index": idx,
                "failure.injected": inject,
                "failure.type": self.config.failure_type.value if inject else "none",
            },
        ) as span:
            rec = InjectionResult(
                tool_name=name,
                failure_type=self.config.failure_type if inject else ToolFailureType.NONE,
                injected=inject,
                call_index=idx,
                span_id=format(span.get_span_context().span_id, "016x"),
            )
            try:
                if inject:
                    exc = self._build_exception()
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    rec.exception = exc
                    raise exc
                return fn(*args, **kwargs)
            except ToolCallError:
                raise
            except Exception as exc:
                # Real (non-injected) error — record it
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                rec.exception = exc
                raise
            finally:
                rec.latency_ms = (time.perf_counter() - start) * 1000
                self._log.append(rec)

    async def _execute_async(self, fn: Callable, name: str, *args: Any, **kwargs: Any) -> Any:
        self._call_count += 1
        idx = self._call_count
        inject = self._should_inject()
        start = time.perf_counter()

        with _tracer.start_as_current_span(
            f"tool.call.{name}",
            attributes={
                "tool.name": name,
                "tool.call_index": idx,
                "failure.injected": inject,
                "failure.type": self.config.failure_type.value if inject else "none",
            },
        ) as span:
            rec = InjectionResult(
                tool_name=name,
                failure_type=self.config.failure_type if inject else ToolFailureType.NONE,
                injected=inject,
                call_index=idx,
                span_id=format(span.get_span_context().span_id, "016x"),
            )
            try:
                if inject:
                    if self.config.failure_type == ToolFailureType.TIMEOUT:
                        # Simulate actual async hang before raising
                        await asyncio.sleep(self.config.timeout_seconds)
                    exc = self._build_exception()
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    rec.exception = exc
                    raise exc
                return await fn(*args, **kwargs)
            except ToolCallError:
                raise
            except Exception as exc:
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                rec.exception = exc
                raise
            finally:
                rec.latency_ms = (time.perf_counter() - start) * 1000
                self._log.append(rec)

    def _build_exception(self) -> ToolCallError:
        match self.config.failure_type:
            case ToolFailureType.TIMEOUT:
                return ToolCallTimeout(
                    f"Tool call timed out after {self.config.timeout_seconds}s"
                )
            case ToolFailureType.MALFORMED_OUTPUT:
                payload = self.config.malformed_payload or _random_malformed_payload()
                return ToolCallMalformedOutput(
                    f"Tool returned unparseable output: {payload!r}"
                )
            case ToolFailureType.RATE_LIMIT:
                return ToolCallRateLimited(retry_after=self.config.retry_after_seconds)
            case ToolFailureType.API_ERROR:
                return ToolCallAPIError(status_code=self.config.error_code)
            case _:
                return ToolCallError("Unknown tool failure type")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_MALFORMED_PAYLOADS = [
    '{"result": ',          # truncated JSON
    "not json at all",      # plain string
    '{"result": null,}',    # trailing comma (invalid JSON)
    "\x00\x01\x02\xff",    # binary garbage
    "undefined",            # JavaScript undefined leak
    "<html>502 Bad Gateway</html>",  # HTML error page instead of JSON
    "",                     # empty response
]


def _random_malformed_payload() -> str:
    return random.choice(_MALFORMED_PAYLOADS)
