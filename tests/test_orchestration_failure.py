"""Tests for OrchestrationFailureInjector — Category 2: Orchestration Failures."""

from __future__ import annotations

import pytest

from agentprobe.injector.orchestration_failure import (
    InfiniteLoopError,
    MissingHandoffError,
    OrchestrationFailureConfig,
    OrchestrationFailureInjector,
    OrchestrationFailureType,
    StateCorruptionError,
    WrongBranchError,
)


class TestOrchestrationFailureInjector:
    def test_no_injection_by_default(self):
        injector = OrchestrationFailureInjector()

        def node(state: dict) -> dict:
            return {"value": 1}

        wrapped = injector.wrap_node(node)
        assert wrapped({}) == {"value": 1}

    def test_infinite_loop_raises_after_max_iterations(self):
        cfg = OrchestrationFailureConfig(
            failure_type=OrchestrationFailureType.INFINITE_LOOP,
            max_loop_iterations=3,
        )
        injector = OrchestrationFailureInjector(cfg)

        def node(state: dict) -> dict:
            return {"value": "ok"}

        wrapped = injector.wrap_node(node, node_name="test_node")

        wrapped({})  # call 1 — passes
        wrapped({})  # call 2 — passes
        with pytest.raises(InfiniteLoopError):
            wrapped({})  # call 3 — raises

    def test_infinite_loop_resets_counter_after_raise(self):
        cfg = OrchestrationFailureConfig(
            failure_type=OrchestrationFailureType.INFINITE_LOOP,
            max_loop_iterations=2,
        )
        injector = OrchestrationFailureInjector(cfg)

        def node(state: dict) -> dict:
            return {}

        wrapped = injector.wrap_node(node, node_name="n")
        wrapped({})
        with pytest.raises(InfiniteLoopError):
            wrapped({})

        # Counter should reset — first call after reset should pass
        wrapped({})

    def test_wrong_branch_router_returns_injected_target(self):
        cfg = OrchestrationFailureConfig(
            failure_type=OrchestrationFailureType.WRONG_BRANCH,
            wrong_branch_target="__end__",
        )
        injector = OrchestrationFailureInjector(cfg)

        def real_router(state: dict) -> str:
            return "continue"

        wrapped = injector.wrap_router(real_router)
        assert wrapped({}) == "__end__"

    def test_wrong_branch_default_target_is_end(self):
        cfg = OrchestrationFailureConfig(
            failure_type=OrchestrationFailureType.WRONG_BRANCH,
        )
        injector = OrchestrationFailureInjector(cfg)

        def router(state: dict) -> str:
            return "tools"

        wrapped = injector.wrap_router(router)
        assert wrapped({}) == "__end__"

    def test_state_corruption_reverses_list(self):
        cfg = OrchestrationFailureConfig(
            failure_type=OrchestrationFailureType.STATE_CORRUPTION,
            corrupt_keys=["messages"],
        )
        injector = OrchestrationFailureInjector(cfg)

        def node(state: dict) -> dict:
            return {"messages": ["hello", "world"]}

        wrapped = injector.wrap_node(node)
        result = wrapped({})
        assert result["messages"] == ["world", "hello"]

    def test_state_corruption_negates_int(self):
        cfg = OrchestrationFailureConfig(
            failure_type=OrchestrationFailureType.STATE_CORRUPTION,
            corrupt_keys=["count"],
        )
        injector = OrchestrationFailureInjector(cfg)

        def node(state: dict) -> dict:
            return {"count": 5}

        wrapped = injector.wrap_node(node)
        result = wrapped({})
        assert result["count"] == -6  # -value - 1

    def test_missing_handoff_raises(self):
        cfg = OrchestrationFailureConfig(
            failure_type=OrchestrationFailureType.MISSING_HANDOFF,
        )
        injector = OrchestrationFailureInjector(cfg)

        def node(state: dict) -> dict:
            return {"output": "result"}

        wrapped = injector.wrap_node(node, node_name="handoff_node")
        with pytest.raises(MissingHandoffError):
            wrapped({})

    def test_injection_log_populated(self):
        cfg = OrchestrationFailureConfig(
            failure_type=OrchestrationFailureType.WRONG_BRANCH,
        )
        injector = OrchestrationFailureInjector(cfg)

        def router(state: dict) -> str:
            return "tools"

        wrapped = injector.wrap_router(router)
        for _ in range(3):
            wrapped({})

        assert len(injector.injection_log) == 3
        assert all(r.injected for r in injector.injection_log)

    def test_reset_clears_state(self):
        cfg = OrchestrationFailureConfig(
            failure_type=OrchestrationFailureType.INFINITE_LOOP,
            max_loop_iterations=2,
        )
        injector = OrchestrationFailureInjector(cfg)

        def node(state: dict) -> dict:
            return {}

        wrapped = injector.wrap_node(node, node_name="n")
        wrapped({})  # call 1 — passes
        with pytest.raises(InfiniteLoopError):
            wrapped({})  # call 2 — raises (2 >= max_loop_iterations=2)

        injector.reset()
        assert len(injector.injection_log) == 0
        # After reset, loop counter cleared — first call should pass again
        wrapped({})

    def test_summary_counts(self):
        cfg = OrchestrationFailureConfig(
            failure_type=OrchestrationFailureType.WRONG_BRANCH,
            failure_probability=1.0,
        )
        injector = OrchestrationFailureInjector(cfg)

        def router(state: dict) -> str:
            return "tools"

        wrapped = injector.wrap_router(router)
        for _ in range(5):
            wrapped({})

        s = injector.summary()
        assert s["total_node_calls"] == 5
        assert s["injected_failures"] == 5
        assert s["injection_rate"] == 1.0
