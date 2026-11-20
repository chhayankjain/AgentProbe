"""The OTLP exporter uses TLS unless the endpoint or settings say otherwise."""

from __future__ import annotations

import pytest

from agentprobe.observer import tracer as tracer_module
from agentprobe.observer.tracer import otlp_insecure


@pytest.fixture(autouse=True)
def _no_otel_env(monkeypatch):
    for var in ("OTEL_EXPORTER_OTLP_TRACES_INSECURE", "OTEL_EXPORTER_OTLP_INSECURE"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("https://collector.example.com:4317", False),  # TLS
        ("HTTPS://collector.example.com:4317", False),
        ("http://localhost:4317", True),  # plaintext, as before
        ("localhost:4317", True),  # no scheme: plaintext, as in 1.4
    ],
)
def test_default_follows_the_endpoint(endpoint, expected):
    assert otlp_insecure(endpoint) is expected


def test_explicit_choice_wins():
    assert otlp_insecure("http://localhost:4317", insecure=False) is False
    assert otlp_insecure("https://collector:4317", insecure=True) is True


@pytest.mark.parametrize(
    "var", ["OTEL_EXPORTER_OTLP_INSECURE", "OTEL_EXPORTER_OTLP_TRACES_INSECURE"]
)
def test_standard_environment_variables(monkeypatch, var):
    monkeypatch.setenv(var, "false")
    assert otlp_insecure("http://localhost:4317") is False
    assert otlp_insecure("localhost:4317") is False
    monkeypatch.setenv(var, "TRUE")
    assert otlp_insecure("localhost:4317") is True


def test_https_is_never_downgraded_by_the_environment(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_INSECURE", "true")
    assert otlp_insecure("https://collector:4317") is False


def test_traces_variable_takes_precedence(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_INSECURE", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_INSECURE", "false")
    assert otlp_insecure("localhost:4317") is False


def test_provider_passes_the_choice_to_the_exporter(monkeypatch):
    if not tracer_module._OTLP_AVAILABLE:
        pytest.skip("OTLP exporter not installed")
    seen: list[dict] = []

    class FakeExporter:
        def __init__(self, **kwargs):
            seen.append(kwargs)

        def export(self, spans):  # pragma: no cover - never flushed here
            return None

        def shutdown(self):
            pass

        def force_flush(self, timeout_millis=None):
            return True

    monkeypatch.setattr(tracer_module, "OTLPSpanExporter", FakeExporter)
    tracer_module._build_provider("svc", "https://collector:4317", console=False)
    tracer_module._build_provider("svc", "http://localhost:4317", console=False)
    tracer_module._build_provider("svc", "http://localhost:4317", console=False, insecure=False)
    assert [k["insecure"] for k in seen] == [False, True, False]
