"""Latency failure injection for agentic LLM systems.

Failure taxonomy — Latency Failures (Category 4):
  - P99_SPIKE       : Inject a single large latency spike (tail latency simulation)
  - CASCADING_SLOW  : Each successive call gets progressively slower
  - COLD_START      : First call has large startup delay, subsequent calls are normal
  - JITTER          : Add random jitter to every call (simulates network instability)

Unlike tool/orchestration failures, latency injectors do NOT raise exceptions.
They slow down execution to expose how agent frameworks handle slow responses,
impact downstream P99 latency, and affect user-perceived performance.

Usage::

    injector = LatencyFailureInjector(
        config=LatencyFailureConfig(
            failure_type=LatencyFailureType.P99_SPIKE,
            spike_delay_seconds=5.0,
            spike_probability=0.01,
        )
    )

    @injector.wrap
    def call_llm(prompt: str) -> str:
        return llm_api.complete(prompt)
"""

from __future__ import annotations

import asyncio
import functools
import math
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, TypeVar

from agentprobe.observer.tracer import get_tracer

T = TypeVar("T")

_tracer = get_tracer(__name__)


# ---------------------------------------------------------------------------
# Enums and config
# ---------------------------------------------------------------------------


class LatencyFailureType(str, Enum):
    NONE = "none"
    P99_SPIKE = "p99_spike"
    CASCADING_SLOW = "cascading_slow"
    COLD_START = "cold_start"
    JITTER = "jitter"


@dataclass
class LatencyFailureConfig:
    """Configuration for latency failure injection.

    Attributes:
        failure_type: Which latency pattern to inject.
        spike_delay_seconds: Delay to inject for P99_SPIKE mode.
        spike_probability: Probability of a spike on each call (P99_SPIKE).
            Set to 0.01 to simulate ~P99 tail latency.
        cascade_increment_seconds: Additional delay added per call (CASCADING_SLOW).
        cascade_max_seconds: Maximum total delay cap (CASCADING_SLOW).
        cold_start_delay_seconds: First-call delay (COLD_START).
        jitter_min_seconds: Minimum jitter per call (JITTER).
        jitter_max_seconds: Maximum jitter per call (JITTER).
        seed: RNG seed for reproducible experiments.
    """

    failure_type: LatencyFailureType = LatencyFailureType.NONE
    spike_delay_seconds: float = 5.0
    spike_probability: float = 0.01
    cascade_increment_seconds: float = 0.5
    cascade_max_seconds: float = 10.0
    cold_start_delay_seconds: float = 3.0
    jitter_min_seconds: float = 0.1
    jitter_max_seconds: float = 2.0
    seed: int | None = None


# ---------------------------------------------------------------------------
# Latency record
# ---------------------------------------------------------------------------


@dataclass
class LatencyRecord:
    call_index: int
    base_latency_ms: float
    injected_delay_ms: float
    total_latency_ms: float
    failure_type: LatencyFailureType
    was_spike: bool = False
    was_cold_start: bool = False


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------


class LatencyFailureInjector:
    """Injects latency failures into agent function calls.

    Does not raise exceptions — instead adds synchronous or async delays
    to simulate real-world latency distributions including tail latency.
    """

    def __init__(self, config: LatencyFailureConfig | None = None) -> None:
        self.config = config or LatencyFailureConfig()
        self._rng = random.Random(self.config.seed)
        self._log: list[LatencyRecord] = []
        self._call_count = 0
        self._cold_start_done = False
        self._cascade_delay = 0.0

    # ------------------------------------------------------------------
    # Decorator API
    # ------------------------------------------------------------------

    def wrap(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Wrap a synchronous function with latency injection."""

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self._execute_sync(fn, *args, **kwargs)  # type: ignore[return-value]

        return wrapper  # type: ignore[return-value]

    def wrap_async(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Wrap an async function with latency injection."""

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await self._execute_async(fn, *args, **kwargs)  # type: ignore[return-value]

        return wrapper  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    @property
    def latency_log(self) -> list[LatencyRecord]:
        return list(self._log)

    def reset(self) -> None:
        self._log.clear()
        self._call_count = 0
        self._cold_start_done = False
        self._cascade_delay = 0.0
        self._rng = random.Random(self.config.seed)

    def summary(self) -> dict[str, Any]:
        if not self._log:
            return {"total_calls": 0}
        latencies = [r.total_latency_ms for r in self._log]
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        return {
            "total_calls": n,
            "mean_latency_ms": sum(sorted_lat) / n,
            "p50_latency_ms": sorted_lat[int(n * 0.50)],
            "p90_latency_ms": sorted_lat[int(n * 0.90)],
            "p99_latency_ms": sorted_lat[min(int(n * 0.99), n - 1)],
            "max_latency_ms": sorted_lat[-1],
            "spikes": sum(1 for r in self._log if r.was_spike),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_delay(self) -> tuple[float, bool, bool]:
        """Return (delay_seconds, was_spike, was_cold_start)."""
        cfg = self.config
        match cfg.failure_type:
            case LatencyFailureType.NONE:
                return 0.0, False, False

            case LatencyFailureType.P99_SPIKE:
                is_spike = self._rng.random() < cfg.spike_probability
                return (cfg.spike_delay_seconds if is_spike else 0.0), is_spike, False

            case LatencyFailureType.CASCADING_SLOW:
                self._cascade_delay = min(
                    self._cascade_delay + cfg.cascade_increment_seconds,
                    cfg.cascade_max_seconds,
                )
                return self._cascade_delay, False, False

            case LatencyFailureType.COLD_START:
                if not self._cold_start_done:
                    self._cold_start_done = True
                    return cfg.cold_start_delay_seconds, False, True
                return 0.0, False, False

            case LatencyFailureType.JITTER:
                jitter = self._rng.uniform(cfg.jitter_min_seconds, cfg.jitter_max_seconds)
                return jitter, False, False

            case _:
                return 0.0, False, False

    def _execute_sync(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        self._call_count += 1
        delay, was_spike, was_cold = self._compute_delay()

        with _tracer.start_as_current_span(
            f"latency.call.{fn.__name__}",
            attributes={
                "latency.injected_delay_ms": delay * 1000,
                "latency.failure_type": self.config.failure_type.value,
                "latency.was_spike": was_spike,
                "latency.was_cold_start": was_cold,
            },
        ):
            start = time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            fn_start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                end = time.perf_counter()
                self._log.append(LatencyRecord(
                    call_index=self._call_count,
                    base_latency_ms=(end - fn_start) * 1000,
                    injected_delay_ms=delay * 1000,
                    total_latency_ms=(end - start) * 1000,
                    failure_type=self.config.failure_type,
                    was_spike=was_spike,
                    was_cold_start=was_cold,
                ))

    async def _execute_async(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        self._call_count += 1
        delay, was_spike, was_cold = self._compute_delay()

        with _tracer.start_as_current_span(
            f"latency.call.{fn.__name__}",
            attributes={
                "latency.injected_delay_ms": delay * 1000,
                "latency.failure_type": self.config.failure_type.value,
                "latency.was_spike": was_spike,
                "latency.was_cold_start": was_cold,
            },
        ):
            start = time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            fn_start = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                end = time.perf_counter()
                self._log.append(LatencyRecord(
                    call_index=self._call_count,
                    base_latency_ms=(end - fn_start) * 1000,
                    injected_delay_ms=delay * 1000,
                    total_latency_ms=(end - start) * 1000,
                    failure_type=self.config.failure_type,
                    was_spike=was_spike,
                    was_cold_start=was_cold,
                ))
