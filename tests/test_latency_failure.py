"""Tests for LatencyFailureInjector — Category 4: Latency Failures."""

from __future__ import annotations

import pytest

from agentprobe.injector.latency_failure import (
    LatencyFailureConfig,
    LatencyFailureInjector,
    LatencyFailureType,
)


class TestLatencyFailureInjector:
    def test_no_injection_by_default(self):
        injector = LatencyFailureInjector()

        @injector.wrap
        def fn() -> str:
            return "ok"

        assert fn() == "ok"
        assert injector.latency_log[0].injected_delay_ms == 0.0

    def test_does_not_raise_exceptions(self):
        """Latency injector adds delay but never raises."""
        cfg = LatencyFailureConfig(
            failure_type=LatencyFailureType.P99_SPIKE,
            spike_delay_seconds=0.001,
            spike_probability=1.0,
        )
        injector = LatencyFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        assert fn() == "ok"  # No exception raised

    def test_cold_start_only_delays_first_call(self):
        cfg = LatencyFailureConfig(
            failure_type=LatencyFailureType.COLD_START,
            cold_start_delay_seconds=0.005,
        )
        injector = LatencyFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        fn()  # cold start
        fn()  # warm
        fn()  # warm

        log = injector.latency_log
        assert log[0].was_cold_start is True
        assert log[1].was_cold_start is False
        assert log[1].injected_delay_ms == 0.0
        assert log[2].injected_delay_ms == 0.0

    def test_cascading_slow_delay_increases_monotonically(self):
        cfg = LatencyFailureConfig(
            failure_type=LatencyFailureType.CASCADING_SLOW,
            cascade_increment_seconds=0.001,
            cascade_max_seconds=1.0,
        )
        injector = LatencyFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        for _ in range(5):
            fn()

        delays = [r.injected_delay_ms for r in injector.latency_log]
        for i in range(1, len(delays)):
            assert delays[i] >= delays[i - 1]

    def test_cascading_slow_respects_max_cap(self):
        cfg = LatencyFailureConfig(
            failure_type=LatencyFailureType.CASCADING_SLOW,
            cascade_increment_seconds=0.1,
            cascade_max_seconds=0.2,
        )
        injector = LatencyFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        for _ in range(10):
            fn()

        max_delay = max(r.injected_delay_ms for r in injector.latency_log)
        assert max_delay <= 200.0 + 1.0  # 200ms cap with 1ms tolerance

    def test_jitter_all_calls_get_delay(self):
        cfg = LatencyFailureConfig(
            failure_type=LatencyFailureType.JITTER,
            jitter_min_seconds=0.001,
            jitter_max_seconds=0.005,
            seed=42,
        )
        injector = LatencyFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        for _ in range(10):
            fn()

        delays = [r.injected_delay_ms for r in injector.latency_log]
        assert all(d >= 1.0 for d in delays)   # min 1ms
        assert all(d <= 10.0 for d in delays)  # max ~5ms + tolerance

    def test_p99_spike_marks_spike_calls(self):
        cfg = LatencyFailureConfig(
            failure_type=LatencyFailureType.P99_SPIKE,
            spike_delay_seconds=0.001,
            spike_probability=1.0,  # 100% spike rate for testing
        )
        injector = LatencyFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        for _ in range(5):
            fn()

        spikes = [r for r in injector.latency_log if r.was_spike]
        assert len(spikes) == 5

    def test_summary_percentiles_ordered(self):
        cfg = LatencyFailureConfig(
            failure_type=LatencyFailureType.JITTER,
            jitter_min_seconds=0.001,
            jitter_max_seconds=0.01,
            seed=42,
        )
        injector = LatencyFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        for _ in range(50):
            fn()

        s = injector.summary()
        assert s["total_calls"] == 50
        assert s["p50_latency_ms"] <= s["p90_latency_ms"]
        assert s["p90_latency_ms"] <= s["p99_latency_ms"]
        assert s["p99_latency_ms"] <= s["max_latency_ms"]

    def test_reset_clears_log_and_state(self):
        cfg = LatencyFailureConfig(
            failure_type=LatencyFailureType.COLD_START,
            cold_start_delay_seconds=0.001,
        )
        injector = LatencyFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        fn()
        assert injector.latency_log[0].was_cold_start is True

        injector.reset()
        assert len(injector.latency_log) == 0

        fn()
        # After reset, cold start fires again
        assert injector.latency_log[0].was_cold_start is True

    def test_none_type_produces_zero_delay(self):
        cfg = LatencyFailureConfig(failure_type=LatencyFailureType.NONE)
        injector = LatencyFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        for _ in range(5):
            fn()

        assert all(r.injected_delay_ms == 0.0 for r in injector.latency_log)
