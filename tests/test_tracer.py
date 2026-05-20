"""Tests for AgentProbeTracer and get_tracer utility."""

from __future__ import annotations

from agentprobe.observer.tracer import AgentProbeTracer, get_tracer


class TestAgentProbeTracer:
    def test_creates_without_otlp(self):
        tracer = AgentProbeTracer(
            service_name="test-service",
            otlp_endpoint=None,
            console=False,
        )
        assert tracer.service_name == "test-service"
        tracer.shutdown()

    def test_span_context_manager(self):
        tracer = AgentProbeTracer(service_name="test", console=False)
        with tracer.span("test.operation", attributes={"key": "value"}) as span:
            assert span is not None
        tracer.shutdown()

    def test_span_without_attributes(self):
        tracer = AgentProbeTracer(service_name="test", console=False)
        with tracer.span("test.bare") as span:
            assert span is not None
        tracer.shutdown()

    def test_get_tracer_module_level(self):
        tracer = AgentProbeTracer(service_name="test", console=False)
        t = get_tracer("test.module")
        assert t is not None
        tracer.shutdown()

    def test_get_tracer_without_provider(self):
        # Should return a no-op tracer without raising
        t = get_tracer("standalone.module")
        assert t is not None

    def test_nested_spans(self):
        tracer = AgentProbeTracer(service_name="test", console=False)
        with tracer.span("parent") as parent:
            with tracer.span("child", attributes={"depth": 1}) as child:
                assert child is not None
            assert parent is not None
        tracer.shutdown()
