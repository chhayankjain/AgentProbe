"""AgentProbe: fault-injection testing for tool-using LLM agents.

Test your own agent with :func:`agentprobe.testing.probe`. It has no required
dependencies::

    from agentprobe.testing import probe

The research toolkit (benchmark runner, injectors, tracer, failure classifier)
needs optional extras and loads only when you use it::

    pip install "agentprobe-toolkit[research]"

    from agentprobe import BenchmarkRunner, ToolFailureInjector

Framework adapters live in :mod:`agentprobe.agents` and are imported explicitly
(``from agentprobe.agents.langgraph_agent import LangGraphToolAgent``).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from agentprobe.probe import Probe, ProbeGateError, ProbeResult

__version__ = "1.4.0"

#: Names loaded on first use, so ``import agentprobe`` stays dependency-free.
#: A missing dependency raises ImportError naming the extra to install.
_LAZY: dict[str, str] = {
    "ToolFailureConfig": "agentprobe.injector.tool_failure",
    "ToolFailureInjector": "agentprobe.injector.tool_failure",
    "ToolFailureType": "agentprobe.injector.tool_failure",
    "AgentProbeTracer": "agentprobe.observer.tracer",
    "get_tracer": "agentprobe.observer.tracer",
    "FailureClassifier": "agentprobe.observer.classifier",
    "FailureCategory": "agentprobe.observer.classifier",
    "BenchmarkConfig": "agentprobe.benchmark.runner",
    "BenchmarkRunner": "agentprobe.benchmark.runner",
    "RunResult": "agentprobe.benchmark.runner",
    "MetricsCalculator": "agentprobe.benchmark.metrics",
    "AgentMetrics": "agentprobe.benchmark.metrics",
}

if TYPE_CHECKING:
    from agentprobe.benchmark.metrics import AgentMetrics, MetricsCalculator
    from agentprobe.benchmark.runner import BenchmarkConfig, BenchmarkRunner, RunResult
    from agentprobe.injector.tool_failure import (
        ToolFailureConfig,
        ToolFailureInjector,
        ToolFailureType,
    )
    from agentprobe.observer.classifier import FailureCategory, FailureClassifier
    from agentprobe.observer.tracer import AgentProbeTracer, get_tracer


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'agentprobe' has no attribute {name!r}")
    try:
        value = getattr(importlib.import_module(module), name)
    except ModuleNotFoundError as exc:
        from agentprobe._extras import missing_extra

        raise missing_extra(exc, f"agentprobe.{name}") from exc
    globals()[name] = value  # cache: later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "__version__",
    # Test your own agent (the probe() helper lives in agentprobe.testing)
    "Probe",
    "ProbeResult",
    "ProbeGateError",
    # Injection
    "ToolFailureConfig",
    "ToolFailureInjector",
    "ToolFailureType",
    # Observability
    "AgentProbeTracer",
    "get_tracer",
    "FailureClassifier",
    "FailureCategory",
    # Benchmark
    "BenchmarkConfig",
    "BenchmarkRunner",
    "RunResult",
    "MetricsCalculator",
    "AgentMetrics",
]
