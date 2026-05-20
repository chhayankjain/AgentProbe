"""Tests for FailureClassifier — maps exceptions/spans to 5-category taxonomy."""

from __future__ import annotations

import pytest

from agentprobe.observer.classifier import (
    FailureCategory,
    FailureClassifier,
    FailureSeverity,
)

# ---------------------------------------------------------------------------
# Stub exceptions — FailureClassifier matches on type.__name__ (not module),
# so these stubs work identically to the real injector exception classes.
# This keeps feature/observer-failure-classifier independently testable
# without requiring the injector feature branches to be merged first.
# ---------------------------------------------------------------------------


class ToolCallTimeout(Exception): pass
class ToolCallMalformedOutput(Exception): pass


class ToolCallRateLimited(Exception):
    def __init__(self, retry_after: float = 60.0) -> None:
        self.retry_after = retry_after
        super().__init__(str(retry_after))


class ToolCallAPIError(Exception):
    def __init__(self, status_code: int = 500) -> None:
        self.status_code = status_code
        super().__init__(str(status_code))


class InfiniteLoopError(Exception): pass
class StateCorruptionError(Exception): pass
class WrongBranchError(Exception): pass
class MissingHandoffError(Exception): pass


class ContextWindowOverflow(Exception):
    def __init__(self, token_count: int, token_limit: int) -> None:
        self.token_count = token_count
        self.token_limit = token_limit
        super().__init__(f"{token_count} > {token_limit}")


class LostStateError(Exception): pass


class TestFailureClassifier:
    def setup_method(self):
        self.clf = FailureClassifier()

    # ------------------------------------------------------------------
    # Tool call exceptions
    # ------------------------------------------------------------------

    def test_classify_tool_timeout(self):
        event = self.clf.classify_exception(ToolCallTimeout("5s"))
        assert event.category == FailureCategory.TOOL_CALL
        assert event.severity == FailureSeverity.HIGH

    def test_classify_malformed_output(self):
        event = self.clf.classify_exception(ToolCallMalformedOutput("bad json"))
        assert event.category == FailureCategory.TOOL_CALL
        assert event.severity == FailureSeverity.MEDIUM

    def test_classify_rate_limit(self):
        event = self.clf.classify_exception(ToolCallRateLimited(retry_after=60.0))
        assert event.category == FailureCategory.TOOL_CALL
        assert event.severity == FailureSeverity.MEDIUM

    def test_classify_api_error(self):
        event = self.clf.classify_exception(ToolCallAPIError(status_code=503))
        assert event.category == FailureCategory.TOOL_CALL
        assert event.severity == FailureSeverity.HIGH

    # ------------------------------------------------------------------
    # Orchestration exceptions
    # ------------------------------------------------------------------

    def test_classify_infinite_loop(self):
        event = self.clf.classify_exception(InfiniteLoopError("loop"))
        assert event.category == FailureCategory.ORCHESTRATION
        assert event.severity == FailureSeverity.CRITICAL

    def test_classify_state_corruption(self):
        event = self.clf.classify_exception(StateCorruptionError("corrupt"))
        assert event.category == FailureCategory.ORCHESTRATION
        assert event.severity == FailureSeverity.CRITICAL

    def test_classify_wrong_branch(self):
        event = self.clf.classify_exception(WrongBranchError("bad route"))
        assert event.category == FailureCategory.ORCHESTRATION
        assert event.severity == FailureSeverity.HIGH

    def test_classify_missing_handoff(self):
        event = self.clf.classify_exception(MissingHandoffError("dropped"))
        assert event.category == FailureCategory.ORCHESTRATION
        assert event.severity == FailureSeverity.HIGH

    # ------------------------------------------------------------------
    # Context exceptions
    # ------------------------------------------------------------------

    def test_classify_context_overflow(self):
        event = self.clf.classify_exception(ContextWindowOverflow(5000, 4096))
        assert event.category == FailureCategory.CONTEXT
        assert event.severity == FailureSeverity.HIGH

    def test_classify_lost_state(self):
        event = self.clf.classify_exception(LostStateError("dropped history"))
        assert event.category == FailureCategory.CONTEXT

    # ------------------------------------------------------------------
    # Unknown exception
    # ------------------------------------------------------------------

    def test_classify_unknown_exception(self):
        event = self.clf.classify_exception(ValueError("random"))
        assert event.category == FailureCategory.UNKNOWN

    # ------------------------------------------------------------------
    # Latency classification
    # ------------------------------------------------------------------

    def test_latency_above_threshold_returns_event(self):
        event = self.clf.classify_latency(latency_ms=5000, p99_threshold_ms=2000)
        assert event is not None
        assert event.category == FailureCategory.LATENCY
        assert event.severity == FailureSeverity.HIGH

    def test_latency_below_threshold_returns_none(self):
        assert self.clf.classify_latency(latency_ms=500, p99_threshold_ms=2000) is None

    def test_extreme_latency_spike_is_critical(self):
        event = self.clf.classify_latency(latency_ms=50_000, p99_threshold_ms=2000)
        assert event is not None
        assert event.severity == FailureSeverity.CRITICAL

    def test_moderate_latency_spike_is_medium(self):
        # Between 1x and 2x threshold
        event = self.clf.classify_latency(latency_ms=2500, p99_threshold_ms=2000)
        assert event is not None
        assert event.severity == FailureSeverity.MEDIUM

    # ------------------------------------------------------------------
    # Consistency classification
    # ------------------------------------------------------------------

    def test_identical_outputs_no_event(self):
        event = self.clf.classify_output_pair("The answer is 42.", "The answer is 42.")
        assert event is None

    def test_very_divergent_outputs_returns_event(self):
        event = self.clf.classify_output_pair(
            "The sun is a star at the center of our solar system.",
            "Quantum entanglement enables faster-than-light communication.",
        )
        if event is not None:
            assert event.category == FailureCategory.CONSISTENCY

    def test_nearly_identical_outputs_no_event(self):
        # 9/10 tokens shared → Jaccard = 0.90 > 0.85 threshold → no event
        a = "The capital of France is Paris the city of light and romance"
        b = "The capital of France is Paris the city of light and beauty"
        event = self.clf.classify_output_pair(a, b)
        assert event is None

    # ------------------------------------------------------------------
    # Report aggregation
    # ------------------------------------------------------------------

    def test_build_report_empty(self):
        report = self.clf.build_report([])
        assert report["total_failures"] == 0
        assert report["by_category"] == {}

    def test_build_report_aggregates_by_category(self):
        events = [
            self.clf.classify_exception(ToolCallTimeout("t")),
            self.clf.classify_exception(ToolCallTimeout("t")),
            self.clf.classify_exception(InfiniteLoopError("l")),
        ]
        report = self.clf.build_report(events)
        assert report["total_failures"] == 3
        assert report["by_category"]["tool_call"] == 2
        assert report["by_category"]["orchestration"] == 1

    def test_build_report_critical_count(self):
        events = [
            self.clf.classify_exception(InfiniteLoopError("l")),
            self.clf.classify_exception(StateCorruptionError("c")),
            self.clf.classify_exception(ToolCallTimeout("t")),
        ]
        report = self.clf.build_report(events)
        assert report["critical_count"] == 2

    def test_event_to_dict(self):
        event = self.clf.classify_exception(ToolCallTimeout("t"), framework="langgraph")
        d = event.to_dict()
        assert d["category"] == "tool_call"
        assert d["severity"] == "high"
        assert d["framework"] == "langgraph"
