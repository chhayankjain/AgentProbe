"""Failure classification for AgentProbe.

The classifier takes span attributes, exception types, and log records
and maps them to the 5-category AgentProbe failure taxonomy:

  1. TOOL_CALL       — tool_failure.py exceptions / tool span errors
  2. ORCHESTRATION   — orchestration_failure.py exceptions / routing errors
  3. CONTEXT         — context_failure.py exceptions / token overflow
  4. LATENCY         — high-latency spans / tail latency detection
  5. CONSISTENCY     — non-deterministic output patterns / divergent state

Usage::

    classifier = FailureClassifier()

    # From an exception
    event = classifier.classify_exception(exc, context={"tool_name": "search"})

    # From a span (OpenTelemetry SpanData)
    events = classifier.classify_spans(finished_spans)

    # From a latency measurement
    event = classifier.classify_latency(latency_ms=4800, p99_threshold_ms=2000)

    # Batch classification from benchmark run output
    report = classifier.build_report(events)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


class FailureCategory(str, Enum):
    TOOL_CALL = "tool_call"
    ORCHESTRATION = "orchestration"
    CONTEXT = "context"
    LATENCY = "latency"
    CONSISTENCY = "consistency"
    UNKNOWN = "unknown"


class FailureSeverity(str, Enum):
    LOW = "low"        # Transient, likely self-healing
    MEDIUM = "medium"  # Impacts single agent turn
    HIGH = "high"      # Impacts full task completion
    CRITICAL = "critical"  # Causes cascading failures


# ---------------------------------------------------------------------------
# Failure event
# ---------------------------------------------------------------------------


@dataclass
class FailureEvent:
    """A classified failure event from an agent execution."""

    category: FailureCategory
    severity: FailureSeverity
    description: str
    timestamp: float = field(default_factory=time.time)
    framework: str = "unknown"
    task_id: str = ""
    span_id: str = ""
    exception_type: str = ""
    raw_attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "description": self.description,
            "timestamp": self.timestamp,
            "framework": self.framework,
            "task_id": self.task_id,
            "span_id": self.span_id,
            "exception_type": self.exception_type,
        }


# ---------------------------------------------------------------------------
# Classification rules
# ---------------------------------------------------------------------------

# Maps exception type names (or prefixes) to (category, severity)
_EXCEPTION_RULES: list[tuple[str, FailureCategory, FailureSeverity]] = [
    # Tool failures
    ("ToolCallTimeout", FailureCategory.TOOL_CALL, FailureSeverity.HIGH),
    ("ToolCallMalformedOutput", FailureCategory.TOOL_CALL, FailureSeverity.MEDIUM),
    ("ToolCallRateLimited", FailureCategory.TOOL_CALL, FailureSeverity.MEDIUM),
    ("ToolCallAPIError", FailureCategory.TOOL_CALL, FailureSeverity.HIGH),
    ("ToolCallError", FailureCategory.TOOL_CALL, FailureSeverity.MEDIUM),
    # Orchestration failures
    ("InfiniteLoopError", FailureCategory.ORCHESTRATION, FailureSeverity.CRITICAL),
    ("WrongBranchError", FailureCategory.ORCHESTRATION, FailureSeverity.HIGH),
    ("StateCorruptionError", FailureCategory.ORCHESTRATION, FailureSeverity.CRITICAL),
    ("MissingHandoffError", FailureCategory.ORCHESTRATION, FailureSeverity.HIGH),
    ("OrchestrationError", FailureCategory.ORCHESTRATION, FailureSeverity.HIGH),
    # Context failures
    ("ContextWindowOverflow", FailureCategory.CONTEXT, FailureSeverity.HIGH),
    ("LostStateError", FailureCategory.CONTEXT, FailureSeverity.MEDIUM),
    ("HallucinatedHistoryError", FailureCategory.CONTEXT, FailureSeverity.MEDIUM),
    ("ContextError", FailureCategory.CONTEXT, FailureSeverity.MEDIUM),
    # Common LangChain / LLM API exceptions
    ("RateLimitError", FailureCategory.TOOL_CALL, FailureSeverity.MEDIUM),
    ("APITimeoutError", FailureCategory.TOOL_CALL, FailureSeverity.HIGH),
    ("APIConnectionError", FailureCategory.TOOL_CALL, FailureSeverity.HIGH),
    ("BadRequestError", FailureCategory.CONTEXT, FailureSeverity.MEDIUM),
    ("InvalidRequestError", FailureCategory.CONTEXT, FailureSeverity.MEDIUM),
    ("GraphRecursionError", FailureCategory.ORCHESTRATION, FailureSeverity.CRITICAL),
]

# Span attribute keys that signal failure categories
_SPAN_ATTRIBUTE_RULES: dict[str, FailureCategory] = {
    "tool.name": FailureCategory.TOOL_CALL,
    "node.name": FailureCategory.ORCHESTRATION,
    "router.name": FailureCategory.ORCHESTRATION,
    "context.token_estimate": FailureCategory.CONTEXT,
    "latency.injected_delay_ms": FailureCategory.LATENCY,
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class FailureClassifier:
    """Classifies failures into the AgentProbe 5-category taxonomy.

    Stateless — all methods are pure functions over their inputs.
    """

    def classify_exception(
        self,
        exc: Exception,
        context: dict[str, Any] | None = None,
        framework: str = "unknown",
        task_id: str = "",
        span_id: str = "",
    ) -> FailureEvent:
        """Classify a Python exception into a :class:`FailureEvent`."""
        exc_type = type(exc).__name__
        category, severity = self._match_exception(exc_type)
        return FailureEvent(
            category=category,
            severity=severity,
            description=str(exc),
            framework=framework,
            task_id=task_id,
            span_id=span_id,
            exception_type=exc_type,
            raw_attributes=context or {},
        )

    def classify_latency(
        self,
        latency_ms: float,
        p99_threshold_ms: float = 2000.0,
        framework: str = "unknown",
        task_id: str = "",
    ) -> FailureEvent | None:
        """Return a LATENCY FailureEvent if latency exceeds the P99 threshold, else None."""
        if latency_ms <= p99_threshold_ms:
            return None
        severity = (
            FailureSeverity.CRITICAL
            if latency_ms > p99_threshold_ms * 5
            else FailureSeverity.HIGH
            if latency_ms > p99_threshold_ms * 2
            else FailureSeverity.MEDIUM
        )
        return FailureEvent(
            category=FailureCategory.LATENCY,
            severity=severity,
            description=(
                f"Latency spike: {latency_ms:.1f}ms exceeds P99 threshold of "
                f"{p99_threshold_ms:.1f}ms"
            ),
            framework=framework,
            task_id=task_id,
            raw_attributes={"latency_ms": latency_ms, "p99_threshold_ms": p99_threshold_ms},
        )

    def classify_output_pair(
        self,
        output_a: str,
        output_b: str,
        task_id: str = "",
        framework: str = "unknown",
    ) -> FailureEvent | None:
        """Detect consistency failures by comparing two outputs for the same input.

        Returns a CONSISTENCY event if outputs are meaningfully different, else None.
        Uses simple normalized edit-distance heuristic — no LLM calls required.
        """
        if output_a == output_b:
            return None
        similarity = _jaccard_similarity(output_a, output_b)
        if similarity >= 0.85:
            return None  # Close enough — treat as consistent

        severity = FailureSeverity.HIGH if similarity < 0.5 else FailureSeverity.MEDIUM
        return FailureEvent(
            category=FailureCategory.CONSISTENCY,
            severity=severity,
            description=(
                f"Non-deterministic output detected (Jaccard similarity={similarity:.2f})"
            ),
            framework=framework,
            task_id=task_id,
            raw_attributes={"jaccard_similarity": similarity},
        )

    def classify_spans(self, spans: list[Any]) -> list[FailureEvent]:
        """Classify a list of finished OTel SpanData objects into failure events.

        Spans are expected to have .name, .attributes, .status, .events attributes
        (matching the opentelemetry-sdk ReadableSpan interface).
        """
        events: list[FailureEvent] = []
        for span in spans:
            event = self._classify_single_span(span)
            if event:
                events.append(event)
        return events

    def build_report(self, events: list[FailureEvent]) -> dict[str, Any]:
        """Aggregate a list of failure events into a summary report."""
        if not events:
            return {"total_failures": 0, "by_category": {}, "by_severity": {}}

        by_cat: dict[str, int] = {}
        by_sev: dict[str, int] = {}
        for e in events:
            by_cat[e.category.value] = by_cat.get(e.category.value, 0) + 1
            by_sev[e.severity.value] = by_sev.get(e.severity.value, 0) + 1

        return {
            "total_failures": len(events),
            "by_category": by_cat,
            "by_severity": by_sev,
            "critical_count": by_sev.get("critical", 0),
            "frameworks": list({e.framework for e in events}),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _match_exception(
        self, exc_type: str
    ) -> tuple[FailureCategory, FailureSeverity]:
        for prefix, category, severity in _EXCEPTION_RULES:
            if exc_type == prefix or exc_type.startswith(prefix):
                return category, severity
        return FailureCategory.UNKNOWN, FailureSeverity.MEDIUM

    def _classify_single_span(self, span: Any) -> FailureEvent | None:
        """Classify one OTel ReadableSpan. Returns None if no failure detected."""
        from opentelemetry.trace import StatusCode

        if not hasattr(span, "status"):
            return None
        if span.status.status_code != StatusCode.ERROR:
            return None

        attrs = dict(span.attributes or {})
        span_name = getattr(span, "name", "unknown")

        # Determine category from attributes
        category = FailureCategory.UNKNOWN
        for attr_key, cat in _SPAN_ATTRIBUTE_RULES.items():
            if attr_key in attrs:
                category = cat
                break

        # Try to infer from span name prefix
        if category == FailureCategory.UNKNOWN:
            if span_name.startswith("tool."):
                category = FailureCategory.TOOL_CALL
            elif span_name.startswith("orchestration."):
                category = FailureCategory.ORCHESTRATION
            elif span_name.startswith("context."):
                category = FailureCategory.CONTEXT
            elif span_name.startswith("latency."):
                category = FailureCategory.LATENCY

        failure_type = attrs.get("failure.type", "unknown")
        span_id = format(span.context.span_id, "016x") if span.context else ""

        return FailureEvent(
            category=category,
            severity=FailureSeverity.HIGH,
            description=f"Span '{span_name}' failed with type '{failure_type}'",
            span_id=span_id,
            raw_attributes=attrs,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)
