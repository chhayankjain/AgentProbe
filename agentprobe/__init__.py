"""AgentProbe — fault injection and observability toolkit for LLM agents.

Usage::

    from agentprobe import ToolFailureConfig, ToolFailureInjector, ToolFailureType
    from agentprobe import AgentProbeTracer, BenchmarkConfig, BenchmarkRunner, MetricsCalculator

Framework adapters live in :mod:`agentprobe.agents` and are imported explicitly
(``from agentprobe.agents.langgraph_agent import LangGraphToolAgent``) so that
importing the package does not require every framework to be installed.
"""

from agentprobe.benchmark.metrics import AgentMetrics, MetricsCalculator
from agentprobe.benchmark.runner import BenchmarkConfig, BenchmarkRunner, RunResult
from agentprobe.injector.tool_failure import (
    ToolFailureConfig,
    ToolFailureInjector,
    ToolFailureType,
)
from agentprobe.observer.classifier import FailureCategory, FailureClassifier
from agentprobe.observer.tracer import AgentProbeTracer, get_tracer

__version__ = "1.2.0"

__all__ = [
    "__version__",
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
