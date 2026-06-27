"""Tests for ContextFailureInjector — Category 3: Context Failures."""

from __future__ import annotations

import pytest

from agentprobe.injector.context_failure import (
    ContextFailureConfig,
    ContextFailureInjector,
    ContextFailureType,
    ContextWindowOverflow,
)


def _msgs(n: int) -> list[dict]:
    return [{"role": "user", "content": f"message {i}"} for i in range(n)]


class TestContextFailureInjector:
    def test_no_injection_by_default(self):
        injector = ContextFailureInjector()
        msgs = _msgs(5)
        assert injector.corrupt_messages(msgs) == msgs

    def test_lost_state_drops_messages(self):
        cfg = ContextFailureConfig(
            failure_type=ContextFailureType.LOST_STATE,
            drop_fraction=0.4,
            seed=42,
        )
        injector = ContextFailureInjector(cfg)
        result = injector.corrupt_messages(_msgs(10))
        assert len(result) < 10

    def test_lost_state_never_drops_all(self):
        cfg = ContextFailureConfig(
            failure_type=ContextFailureType.LOST_STATE,
            drop_fraction=0.9,
            seed=7,
        )
        injector = ContextFailureInjector(cfg)
        # At minimum 1 message should survive (drop of 10 = 9 dropped, 1 left)
        result = injector.corrupt_messages(_msgs(10))
        assert len(result) >= 1

    def test_overflow_raises_context_window_overflow(self):
        cfg = ContextFailureConfig(
            failure_type=ContextFailureType.OVERFLOW,
            token_limit=10,
        )
        injector = ContextFailureInjector(cfg)
        with pytest.raises(ContextWindowOverflow) as exc_info:
            injector.corrupt_messages(_msgs(10))
        assert exc_info.value.token_limit == 10
        assert exc_info.value.token_count > 10

    def test_hallucinated_history_adds_messages(self):
        cfg = ContextFailureConfig(
            failure_type=ContextFailureType.HALLUCINATED_HISTORY,
            inject_n_fake_messages=3,
        )
        injector = ContextFailureInjector(cfg)
        result = injector.corrupt_messages(_msgs(5))
        assert len(result) == 8

    def test_hallucinated_messages_are_marked(self):
        cfg = ContextFailureConfig(
            failure_type=ContextFailureType.HALLUCINATED_HISTORY,
            inject_n_fake_messages=2,
        )
        injector = ContextFailureInjector(cfg)
        result = injector.corrupt_messages(_msgs(5))
        fake = [m for m in result if m.get("_agentprobe_injected")]
        assert len(fake) == 2

    def test_zero_probability_never_injects(self):
        cfg = ContextFailureConfig(
            failure_type=ContextFailureType.LOST_STATE,
            failure_probability=0.0,
            drop_fraction=1.0,
        )
        injector = ContextFailureInjector(cfg)
        msgs = _msgs(5)
        for _ in range(10):
            result = injector.corrupt_messages(msgs)
            assert len(result) == 5

    def test_wrap_memory_retrieval_decorator(self):
        cfg = ContextFailureConfig(
            failure_type=ContextFailureType.LOST_STATE,
            drop_fraction=0.5,
            seed=1,
        )
        injector = ContextFailureInjector(cfg)

        @injector.wrap_memory_retrieval
        def get_history(session_id: str) -> list[dict]:
            return _msgs(10)

        result = get_history("session-1")
        assert len(result) < 10

    def test_injection_log_records_calls(self):
        cfg = ContextFailureConfig(
            failure_type=ContextFailureType.LOST_STATE,
            drop_fraction=0.3,
        )
        injector = ContextFailureInjector(cfg)
        for _ in range(4):
            injector.corrupt_messages(_msgs(5))
        assert len(injector.injection_log) == 4

    def test_reset_clears_log(self):
        cfg = ContextFailureConfig(failure_type=ContextFailureType.LOST_STATE)
        injector = ContextFailureInjector(cfg)
        injector.corrupt_messages(_msgs(3))
        injector.reset()
        assert len(injector.injection_log) == 0

    def test_invalid_drop_fraction_raises(self):
        with pytest.raises(ValueError):
            ContextFailureConfig(
                failure_type=ContextFailureType.LOST_STATE,
                drop_fraction=1.5,
            )
