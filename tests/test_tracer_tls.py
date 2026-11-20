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


class _FakeExporter:
    created: list[dict] = []

    def __init__(self, **kwargs):
        _FakeExporter.created.append(kwargs)

    def export(self, spans):  # pragma: no cover - never flushed here
        return None

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=None):
        return True


@pytest.fixture
def isolated_provider(monkeypatch):
    """A private provider and registry, so these tests attach nothing global."""
    if not tracer_module._OTLP_AVAILABLE:
        pytest.skip("OTLP exporter not installed")
    from opentelemetry.sdk.trace import TracerProvider

    _FakeExporter.created = []
    monkeypatch.setattr(tracer_module, "OTLPSpanExporter", _FakeExporter)
    monkeypatch.setattr(tracer_module, "_GLOBAL_PROVIDER", TracerProvider())
    monkeypatch.setattr(tracer_module, "_ATTACHED", {"console": True, "otlp": {}})
    return tracer_module


def test_same_endpoint_and_setting_reuses_the_exporter(isolated_provider):
    for _ in range(2):
        isolated_provider.AgentProbeTracer(otlp_endpoint="localhost:4329", insecure=True)
    assert len(_FakeExporter.created) == 1


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        (True, False, "attached without TLS \\(plaintext\\), but this tracer requests TLS"),
        (False, True, "attached with TLS, but this tracer requests plaintext"),
    ],
)
def test_a_conflicting_tls_setting_is_rejected_not_ignored(
    isolated_provider, first, second, message
):
    isolated_provider.AgentProbeTracer(otlp_endpoint="localhost:4329", insecure=first)
    with pytest.raises(ValueError, match=message):
        isolated_provider.AgentProbeTracer(otlp_endpoint="localhost:4329", insecure=second)
    assert [e["insecure"] for e in _FakeExporter.created] == [first]  # no second exporter


def test_the_default_setting_is_compared_after_resolution(isolated_provider):
    """http:// resolves to plaintext, so an explicit insecure=True is the same request."""
    isolated_provider.AgentProbeTracer(otlp_endpoint="http://localhost:4329")
    isolated_provider.AgentProbeTracer(otlp_endpoint="http://localhost:4329", insecure=True)
    with pytest.raises(ValueError):
        isolated_provider.AgentProbeTracer(otlp_endpoint="http://localhost:4329", insecure=False)
    assert len(_FakeExporter.created) == 1


def test_different_endpoints_each_get_their_own_setting(isolated_provider):
    isolated_provider.AgentProbeTracer(otlp_endpoint="localhost:4329", insecure=True)
    isolated_provider.AgentProbeTracer(otlp_endpoint="https://collector:4317")
    assert [e["insecure"] for e in _FakeExporter.created] == [True, False]
