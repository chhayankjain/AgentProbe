"""OpenTelemetry, when it is installed.

``agentprobe.testing`` emits a span per tool call. OpenTelemetry is optional
(the ``observability`` extra): when ``opentelemetry-api`` is importable, spans go
to whatever tracer provider the application configured, exactly as before;
without it, a no-op tracer stands in and recorded span ids are all zeros, the
same value OpenTelemetry reports when no SDK is configured.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class _NoopSpanContext:
    span_id = 0
    trace_id = 0


class _NoopSpan:
    def get_span_context(self) -> _NoopSpanContext:
        return _NoopSpanContext()

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, *args: Any, **kwargs: Any) -> None:
        pass


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any) -> Iterator[_NoopSpan]:
        yield _NoopSpan()


def get_tracer(name: str) -> Any:
    """An OpenTelemetry tracer if OpenTelemetry is installed, otherwise a no-op one."""
    try:
        from opentelemetry import trace
    except ImportError:
        return _NoopTracer()
    return trace.get_tracer(name)
