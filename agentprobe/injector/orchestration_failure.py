"""Orchestration failure injection for agentic LLM systems.

Failure taxonomy — Orchestration Failures (Category 2):
  - INFINITE_LOOP     : Agent cycles through nodes without terminating
  - WRONG_BRANCH      : Agent takes incorrect conditional branch
  - STATE_CORRUPTION  : Agent state dict is silently corrupted mid-execution
  - MISSING_HANDOFF   : Multi-agent handoff message is dropped

These injectors wrap LangGraph StateGraph nodes and edges to simulate
failures in agent control flow rather than individual tool calls.

Usage::

    injector = OrchestrationFailureInjector(
        config=OrchestrationFailureConfig(
            failure_type=OrchestrationFailureType.INFINITE_LOOP,
            max_loop_iterations=3,
        )
    )

    # Wrap a LangGraph node function
    safe_node = injector.wrap_node(my_node_fn, node_name="my_node")

    # Or wrap a conditional edge router
    faulty_router = injector.wrap_router(my_router_fn)
"""

from __future__ import annotations

import functools
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, TypeVar

from opentelemetry.trace import Status, StatusCode

from agentprobe.observer.tracer import get_tracer

T = TypeVar("T")

_tracer = get_tracer(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OrchestrationError(Exception):
    """Base class for injected orchestration errors."""


class InfiniteLoopError(OrchestrationError):
    """Raised when an injected loop exceeds its iteration cap."""


class WrongBranchError(OrchestrationError):
    """Raised to signal incorrect routing in a conditional edge."""


class StateCorruptionError(OrchestrationError):
    """Raised when agent state has been silently corrupted."""


class MissingHandoffError(OrchestrationError):
    """Raised when a multi-agent handoff message is dropped."""


# ---------------------------------------------------------------------------
# Enums and config
# ---------------------------------------------------------------------------


class OrchestrationFailureType(str, Enum):
    NONE = "none"
    INFINITE_LOOP = "infinite_loop"
    WRONG_BRANCH = "wrong_branch"
    STATE_CORRUPTION = "state_corruption"
    MISSING_HANDOFF = "missing_handoff"


@dataclass
class OrchestrationFailureConfig:
    """Configuration for orchestration failure injection.

    Attributes:
        failure_type: Which orchestration failure mode to inject.
        failure_probability: Probability [0, 1] of injecting on each call.
        max_loop_iterations: How many loop iterations before raising (INFINITE_LOOP).
        corrupt_keys: State keys to corrupt (STATE_CORRUPTION).
            If empty, a random key is selected.
        wrong_branch_target: The incorrect branch label to return (WRONG_BRANCH).
            If None, returns "__end__" prematurely.
        seed: RNG seed for reproducible experiments.
    """

    failure_type: OrchestrationFailureType = OrchestrationFailureType.NONE
    failure_probability: float = 1.0
    max_loop_iterations: int = 5
    corrupt_keys: list[str] | None = None
    wrong_branch_target: str | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.failure_probability <= 1.0:
            raise ValueError("failure_probability must be in [0, 1]")


# ---------------------------------------------------------------------------
# Injection result
# ---------------------------------------------------------------------------


@dataclass
class OrchestrationInjectionResult:
    node_name: str
    failure_type: OrchestrationFailureType
    injected: bool
    exception: Exception | None = None
    latency_ms: float = 0.0
    loop_iterations: int = 0
    corrupted_keys: list[str] | None = None


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------


class OrchestrationFailureInjector:
    """Injects orchestration-level failures into LangGraph agent graphs."""

    def __init__(self, config: OrchestrationFailureConfig | None = None) -> None:
        self.config = config or OrchestrationFailureConfig()
        self._rng = random.Random(self.config.seed)
        self._log: list[OrchestrationInjectionResult] = []
        self._loop_counters: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Node wrapping
    # ------------------------------------------------------------------

    def wrap_node(self, fn: Callable, node_name: str = "") -> Callable:
        """Wrap a LangGraph node function with orchestration failure injection.

        The wrapped function accepts the LangGraph state dict and returns
        a (possibly corrupted) state update dict.
        """
        name = node_name or fn.__name__

        @functools.wraps(fn)
        def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            return self._execute_node(fn, name, state)

        return wrapper

    def wrap_router(self, fn: Callable, node_name: str = "") -> Callable:
        """Wrap a LangGraph conditional edge router with wrong-branch injection."""
        name = node_name or fn.__name__

        @functools.wraps(fn)
        def wrapper(state: dict[str, Any]) -> str:
            return self._execute_router(fn, name, state)

        return wrapper

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    @property
    def injection_log(self) -> list[OrchestrationInjectionResult]:
        return list(self._log)

    def reset(self) -> None:
        self._log.clear()
        self._loop_counters.clear()
        self._rng = random.Random(self.config.seed)

    def summary(self) -> dict[str, Any]:
        total = len(self._log)
        injected = sum(1 for r in self._log if r.injected)
        return {
            "total_node_calls": total,
            "injected_failures": injected,
            "injection_rate": injected / total if total > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _should_inject(self) -> bool:
        if self.config.failure_type == OrchestrationFailureType.NONE:
            return False
        return self._rng.random() < self.config.failure_probability

    def _execute_node(
        self, fn: Callable, name: str, state: dict[str, Any]
    ) -> dict[str, Any]:
        inject = self._should_inject()
        start = time.perf_counter()

        with _tracer.start_as_current_span(
            f"orchestration.node.{name}",
            attributes={
                "node.name": name,
                "failure.injected": inject,
                "failure.type": self.config.failure_type.value if inject else "none",
            },
        ) as span:
            rec = OrchestrationInjectionResult(
                node_name=name,
                failure_type=self.config.failure_type if inject else OrchestrationFailureType.NONE,
                injected=inject,
            )
            try:
                if inject:
                    match self.config.failure_type:
                        case OrchestrationFailureType.INFINITE_LOOP:
                            return self._inject_loop(fn, name, state, span, rec)
                        case OrchestrationFailureType.STATE_CORRUPTION:
                            return self._inject_state_corruption(fn, name, state, span, rec)
                        case OrchestrationFailureType.MISSING_HANDOFF:
                            exc = MissingHandoffError(
                                f"Handoff message dropped at node '{name}'"
                            )
                            span.set_status(Status(StatusCode.ERROR, str(exc)))
                            rec.exception = exc
                            raise exc
                        case _:
                            return fn(state)
                return fn(state)
            finally:
                rec.latency_ms = (time.perf_counter() - start) * 1000
                self._log.append(rec)

    def _execute_router(self, fn: Callable, name: str, state: dict[str, Any]) -> str:
        inject = self._should_inject()
        start = time.perf_counter()

        with _tracer.start_as_current_span(
            f"orchestration.router.{name}",
            attributes={
                "router.name": name,
                "failure.injected": inject,
                "failure.type": self.config.failure_type.value if inject else "none",
            },
        ) as span:
            rec = OrchestrationInjectionResult(
                node_name=name,
                failure_type=self.config.failure_type if inject else OrchestrationFailureType.NONE,
                injected=inject,
            )
            try:
                if inject and self.config.failure_type == OrchestrationFailureType.WRONG_BRANCH:
                    wrong = self.config.wrong_branch_target or "__end__"
                    span.set_attribute("router.injected_branch", wrong)
                    span.set_attribute("failure.injected", True)
                    return wrong
                return fn(state)
            finally:
                rec.latency_ms = (time.perf_counter() - start) * 1000
                self._log.append(rec)

    def _inject_loop(
        self,
        fn: Callable,
        name: str,
        state: dict[str, Any],
        span: Any,
        rec: OrchestrationInjectionResult,
    ) -> dict[str, Any]:
        """Run the node repeatedly until max_loop_iterations, then raise."""
        counter = self._loop_counters.get(name, 0) + 1
        self._loop_counters[name] = counter
        rec.loop_iterations = counter

        if counter >= self.config.max_loop_iterations:
            exc = InfiniteLoopError(
                f"Node '{name}' detected as infinite loop after {counter} iterations"
            )
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            rec.exception = exc
            self._loop_counters[name] = 0
            raise exc

        return fn(state)

    def _inject_state_corruption(
        self,
        fn: Callable,
        name: str,
        state: dict[str, Any],
        span: Any,
        rec: OrchestrationInjectionResult,
    ) -> dict[str, Any]:
        """Run the node, then silently corrupt keys in the returned state."""
        result = fn(state)
        keys_to_corrupt = self.config.corrupt_keys or (list(result.keys()) or list(state.keys()))
        if not keys_to_corrupt:
            return result

        # Pick one key to corrupt
        target_key = self._rng.choice(keys_to_corrupt)
        original = result.get(target_key, state.get(target_key))

        corrupted_result = dict(result)
        corrupted_result[target_key] = _corrupt_value(original)

        rec.corrupted_keys = [target_key]
        span.set_attribute("corruption.key", target_key)
        return corrupted_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _corrupt_value(value: Any) -> Any:
    """Return a type-preserving but semantically wrong value."""
    if isinstance(value, str):
        return value[::-1] if value else "CORRUPTED"
    if isinstance(value, int):
        return -value - 1
    if isinstance(value, float):
        return float("nan")
    if isinstance(value, list):
        return list(reversed(value))
    if isinstance(value, dict):
        return {k: None for k in value}
    return None
