"""OpenTelemetry tracing integration for AgentProbe.

Sets up a tracer provider with:
  - Console span exporter (for local dev / CI)
  - OTLP gRPC exporter (for production / Jaeger / Tempo)

Usage::

    from agentprobe.observer.tracer import AgentProbeTracer

    tracer = AgentProbeTracer(
        service_name="my-agent",
        otlp_endpoint="http://localhost:4317",   # optional
    )

    with tracer.span("my.operation", attributes={"key": "value"}) as span:
        result = do_work()
        span.set_attribute("result.count", len(result))

    # Get a module-level tracer (for use inside library code)
    from agentprobe.observer.tracer import get_tracer
    _tracer = get_tracer(__name__)
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Span, Tracer

# Optional OTLP exporter — gracefully unavailable if grpc not installed
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    _OTLP_AVAILABLE = True
except ImportError:
    _OTLP_AVAILABLE = False

_DEFAULT_SERVICE = "agentprobe"
_GLOBAL_PROVIDER: TracerProvider | None = None


def _build_provider(
    service_name: str,
    otlp_endpoint: str | None = None,
    console: bool = True,
) -> TracerProvider:
    resource = Resource(attributes={SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    if console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    if otlp_endpoint and _OTLP_AVAILABLE:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    elif otlp_endpoint and not _OTLP_AVAILABLE:
        import warnings

        warnings.warn(
            "opentelemetry-exporter-otlp-proto-grpc is not installed. "
            "OTLP export is disabled. Install with: "
            "pip install agentprobe[otlp]",
            stacklevel=2,
        )

    return provider


class AgentProbeTracer:
    """Configured OpenTelemetry tracer for AgentProbe experiments.

    Registers as the global OTel tracer provider so that all
    ``get_tracer(__name__)`` calls within the library resolve to this provider.

    Args:
        service_name: Service name shown in trace UIs (Jaeger, Tempo, etc.)
        otlp_endpoint: gRPC endpoint for OTLP export (e.g. "http://localhost:4317").
            If None, only console export is active.
        console: Whether to print spans to stdout (useful for local dev).
    """

    def __init__(
        self,
        service_name: str = _DEFAULT_SERVICE,
        otlp_endpoint: str | None = None,
        console: bool = True,
    ) -> None:
        # Allow env-var override
        endpoint = otlp_endpoint or os.getenv("AGENTPROBE_OTLP_ENDPOINT")
        console_on = console and os.getenv("AGENTPROBE_NO_CONSOLE_TRACE") != "1"

        self._provider = _build_provider(service_name, endpoint, console_on)
        trace.set_tracer_provider(self._provider)
        self._tracer: Tracer = self._provider.get_tracer(service_name)
        self.service_name = service_name

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """Context manager: start a new span with optional attributes."""
        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, v)
            yield span

    def get_tracer(self, name: str) -> Tracer:
        return self._provider.get_tracer(name)

    def shutdown(self) -> None:
        """Flush and shut down the span processor (call at end of experiments)."""
        self._provider.shutdown()


def get_tracer(name: str) -> Tracer:
    """Return a module-level tracer.

    Uses the globally registered OTel provider if :class:`AgentProbeTracer`
    has been initialized; otherwise uses a no-op tracer (safe for unit tests
    that don't need trace output).
    """
    return trace.get_tracer(name)
