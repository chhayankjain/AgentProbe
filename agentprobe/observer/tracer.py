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
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
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
#: "otlp" maps each attached endpoint to whether its exporter is plaintext.
_ATTACHED: dict[str, Any] = {"console": False, "otlp": {}}


def otlp_insecure(endpoint: str, insecure: bool | None = None) -> bool:
    """Whether the OTLP exporter for *endpoint* sends spans without TLS.

    In order: an ``https://`` endpoint uses TLS (OpenTelemetry's exporter always
    does, so ``insecure=True`` with ``https://`` is rejected); otherwise an
    explicit *insecure* wins;
    the standard ``OTEL_EXPORTER_OTLP_TRACES_INSECURE`` or
    ``OTEL_EXPORTER_OTLP_INSECURE`` variable decides if set; an ``http://``
    endpoint is plaintext. An endpoint with no scheme (``localhost:4317``) stays
    plaintext, as in AgentProbe 1.4 and earlier; write ``https://`` to use TLS.
    Certificates come from the standard ``OTEL_EXPORTER_OTLP_CERTIFICATE`` (and
    client certificate/key) variables.
    """
    scheme = urlparse(endpoint).scheme.lower()
    if scheme == "https":
        if insecure:
            raise ValueError(
                f"insecure=True contradicts the https:// endpoint {endpoint!r}: "
                "OpenTelemetry always uses TLS for https://. Use an http:// endpoint "
                "for plaintext, or drop insecure=True."
            )
        return False
    if insecure is not None:
        return insecure
    env = os.getenv("OTEL_EXPORTER_OTLP_TRACES_INSECURE") or os.getenv(
        "OTEL_EXPORTER_OTLP_INSECURE"
    )
    if env is not None and env.strip():
        return env.strip().lower() == "true"
    return True


def _tls_label(insecure: bool) -> str:
    return "without TLS (plaintext)" if insecure else "with TLS"


def _otlp_exporter(endpoint: str, insecure: bool | None) -> Any:
    return OTLPSpanExporter(endpoint=endpoint, insecure=otlp_insecure(endpoint, insecure))


def _build_provider(
    service_name: str,
    otlp_endpoint: str | None = None,
    console: bool = True,
    insecure: bool | None = None,
) -> TracerProvider:
    resource = Resource(attributes={SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    if console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    if otlp_endpoint and _OTLP_AVAILABLE:
        otlp_exporter = _otlp_exporter(otlp_endpoint, insecure)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    elif otlp_endpoint and not _OTLP_AVAILABLE:
        import warnings

        warnings.warn(
            "opentelemetry-exporter-otlp-proto-grpc is not installed. "
            "OTLP export is disabled. Install with: "
            "pip install opentelemetry-exporter-otlp-proto-grpc",
            stacklevel=2,
        )

    return provider


class AgentProbeTracer:
    """Configured OpenTelemetry tracer for AgentProbe experiments.

    The first instance in a process creates the SDK ``TracerProvider`` and
    registers it globally, so every ``get_tracer(__name__)`` in the library
    resolves to it. OpenTelemetry allows the global provider to be set only
    once, so later instances **reuse** that provider instead of creating a
    second one that library spans would never reach; they only attach an OTLP
    exporter for an endpoint that is not attached yet. This makes repeated
    construction (notebooks, ``python -m agentprobe.demo`` in a loop, tests)
    safe.

    Args:
        service_name: Service name shown in trace UIs (Jaeger, Tempo, etc.).
            Only the first instance's name is used for the shared resource.
        otlp_endpoint: gRPC endpoint for OTLP export (e.g. "http://localhost:4317"
            for a local collector, "https://collector.example.com:4317" for TLS).
            If None, only console export is active.
        console: Whether to print spans to stdout (useful for local dev).
        insecure: Send OTLP spans without TLS. ``None`` (the default) decides
            from the endpoint and the standard environment variables; see
            :func:`otlp_insecure`.
    """

    def __init__(
        self,
        service_name: str = _DEFAULT_SERVICE,
        otlp_endpoint: str | None = None,
        console: bool = True,
        insecure: bool | None = None,
    ) -> None:
        global _GLOBAL_PROVIDER
        endpoint = otlp_endpoint or os.getenv("AGENTPROBE_OTLP_ENDPOINT")
        console_on = console and os.getenv("AGENTPROBE_NO_CONSOLE_TRACE") != "1"

        existing = trace.get_tracer_provider()
        if _GLOBAL_PROVIDER is None and isinstance(existing, TracerProvider):
            # The application already configured OpenTelemetry: join it
            # rather than create a provider that library spans would bypass.
            _GLOBAL_PROVIDER = existing
        if _GLOBAL_PROVIDER is None:
            _GLOBAL_PROVIDER = _build_provider(service_name, endpoint, console_on, insecure)
            trace.set_tracer_provider(_GLOBAL_PROVIDER)
            _ATTACHED["console"] = console_on
            if endpoint and _OTLP_AVAILABLE:
                _ATTACHED["otlp"][endpoint] = otlp_insecure(endpoint, insecure)
            self.owns_provider = True
        else:
            self.owns_provider = False
            if console_on and not _ATTACHED["console"]:
                _GLOBAL_PROVIDER.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
                _ATTACHED["console"] = True
            if endpoint and _OTLP_AVAILABLE:
                wanted = otlp_insecure(endpoint, insecure)
                attached = _ATTACHED["otlp"].get(endpoint)
                if attached is None:
                    _GLOBAL_PROVIDER.add_span_processor(
                        BatchSpanProcessor(_otlp_exporter(endpoint, insecure))
                    )
                    _ATTACHED["otlp"][endpoint] = wanted
                elif attached != wanted:
                    # Adding a second exporter would leave the first one sending,
                    # and OpenTelemetry can't remove a span processor: refuse.
                    raise ValueError(
                        f"an OTLP exporter for {endpoint!r} is already attached "
                        f"{_tls_label(attached)}, but this tracer requests "
                        f"{'plaintext' if wanted else 'TLS'}. Use one setting per "
                        "endpoint within a process."
                    )

        self._provider = _GLOBAL_PROVIDER
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

    def add_span_processor(self, processor: Any) -> None:
        """Attach an extra span processor (e.g. an in-memory exporter in tests)."""
        self._provider.add_span_processor(processor)

    def flush(self) -> None:
        """Flush pending spans from every attached exporter."""
        self._provider.force_flush()

    def shutdown(self) -> None:
        """Flush pending spans. The shared provider itself is shut down at
        interpreter exit by the OpenTelemetry SDK, so a tracer going out of
        scope in one experiment never silences spans for the next one."""
        self.flush()


def get_tracer(name: str) -> Tracer:
    """Return a module-level tracer.

    Uses the globally registered OTel provider if :class:`AgentProbeTracer`
    has been initialized; otherwise uses a no-op tracer (safe for unit tests
    that don't need trace output).
    """
    return trace.get_tracer(name)
