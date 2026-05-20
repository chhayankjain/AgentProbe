"""Tests for ToolFailureInjector — Category 1: Tool Call Failures."""

from __future__ import annotations

import pytest

from agentprobe.injector.tool_failure import (
    ToolCallAPIError,
    ToolCallMalformedOutput,
    ToolCallRateLimited,
    ToolCallTimeout,
    ToolFailureConfig,
    ToolFailureInjector,
    ToolFailureType,
)


class TestToolFailureInjector:
    def test_no_injection_by_default(self):
        injector = ToolFailureInjector()

        @injector.wrap
        def fn() -> str:
            return "ok"

        assert fn() == "ok"

    def test_timeout_injection(self):
        cfg = ToolFailureConfig(failure_type=ToolFailureType.TIMEOUT, timeout_seconds=0.0)
        injector = ToolFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        with pytest.raises(ToolCallTimeout):
            fn()

    def test_malformed_output_injection(self):
        cfg = ToolFailureConfig(failure_type=ToolFailureType.MALFORMED_OUTPUT)
        injector = ToolFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        with pytest.raises(ToolCallMalformedOutput):
            fn()

    def test_rate_limit_injection(self):
        cfg = ToolFailureConfig(failure_type=ToolFailureType.RATE_LIMIT, retry_after_seconds=30.0)
        injector = ToolFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        with pytest.raises(ToolCallRateLimited) as exc_info:
            fn()
        assert exc_info.value.retry_after == 30.0

    def test_api_error_injection(self):
        cfg = ToolFailureConfig(failure_type=ToolFailureType.API_ERROR, error_code=503)
        injector = ToolFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        with pytest.raises(ToolCallAPIError) as exc_info:
            fn()
        assert exc_info.value.status_code == 503

    def test_probabilistic_injection(self):
        cfg = ToolFailureConfig(
            failure_type=ToolFailureType.API_ERROR,
            failure_probability=0.5,
            seed=42,
        )
        injector = ToolFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        successes, failures = 0, 0
        for _ in range(100):
            try:
                fn()
                successes += 1
            except ToolCallAPIError:
                failures += 1

        assert 30 <= failures <= 70

    def test_zero_probability_never_injects(self):
        cfg = ToolFailureConfig(
            failure_type=ToolFailureType.TIMEOUT,
            failure_probability=0.0,
        )
        injector = ToolFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        for _ in range(20):
            assert fn() == "ok"

    def test_injection_log_records_all_calls(self):
        cfg = ToolFailureConfig(failure_type=ToolFailureType.API_ERROR, seed=0)
        injector = ToolFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        for _ in range(5):
            try:
                fn()
            except Exception:
                pass

        assert len(injector.injection_log) == 5

    def test_summary_statistics_all_failures(self):
        cfg = ToolFailureConfig(
            failure_type=ToolFailureType.API_ERROR,
            failure_probability=1.0,
        )
        injector = ToolFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        for _ in range(10):
            try:
                fn()
            except Exception:
                pass

        summary = injector.summary()
        assert summary["total_calls"] == 10
        assert summary["injected_failures"] == 10
        assert summary["injection_rate"] == 1.0

    def test_direct_call_method(self):
        cfg = ToolFailureConfig(failure_type=ToolFailureType.MALFORMED_OUTPUT)
        injector = ToolFailureInjector(cfg)

        def fn(x: int) -> int:
            return x * 2

        with pytest.raises(ToolCallMalformedOutput):
            injector.call(fn, 5)

    def test_reset_clears_log_and_counter(self):
        cfg = ToolFailureConfig(failure_type=ToolFailureType.API_ERROR)
        injector = ToolFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        try:
            fn()
        except Exception:
            pass

        assert len(injector.injection_log) == 1
        injector.reset()
        assert len(injector.injection_log) == 0

    def test_invalid_probability_raises_value_error(self):
        with pytest.raises(ValueError):
            ToolFailureConfig(failure_type=ToolFailureType.TIMEOUT, failure_probability=1.5)

    def test_reproducible_with_same_seed(self):
        def _run(seed: int) -> list[bool]:
            cfg = ToolFailureConfig(
                failure_type=ToolFailureType.API_ERROR,
                failure_probability=0.5,
                seed=seed,
            )
            injector = ToolFailureInjector(cfg)

            @injector.wrap
            def fn() -> str:
                return "ok"

            results = []
            for _ in range(20):
                try:
                    fn()
                    results.append(False)
                except ToolCallAPIError:
                    results.append(True)
            return results

        assert _run(99) == _run(99)

    def test_call_index_increments(self):
        cfg = ToolFailureConfig(failure_type=ToolFailureType.NONE)
        injector = ToolFailureInjector(cfg)

        @injector.wrap
        def fn() -> str:
            return "ok"

        for _ in range(5):
            fn()

        indices = [r.call_index for r in injector.injection_log]
        assert indices == [1, 2, 3, 4, 5]
