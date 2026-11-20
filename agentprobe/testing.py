"""pytest-friendly entry point for fault tests on your own agent.

Use :func:`probe` as a context manager in any test::

    from agentprobe.testing import probe

    def test_order_status_survives_a_timeout():
        with probe() as p:
            tools = p.wrap({"payment_status": payment_status})
            p.on("payment_status").fail_first(1, with_="timeout")
            result = p.run(lambda q: answer(q, tools), "Did order 42 go through?")
            result.assert_answer_contains("confirmed")

Or as a fixture in ``conftest.py``::

    import pytest
    from agentprobe.testing import probe as _probe

    @pytest.fixture
    def probe():
        with _probe() as p:
            yield p

When the ``with`` block ends normally, the gate runs: the test fails unless every
required scenario was exercised and every required assertion passed. A full pytest
plugin with markers and reports is planned for v1.5.
"""

from __future__ import annotations

from agentprobe.faults import (
    ToolCallAPIError,
    ToolCallError,
    ToolCallMalformedOutput,
    ToolCallRateLimited,
    ToolCallTimeout,
)
from agentprobe.probe import (
    SCHEMA_VERSION,
    AssertionRecord,
    CallRecord,
    GateReport,
    Probe,
    ProbeError,
    ProbeGateError,
    ProbeResult,
    ScenarioEvidence,
    ToolScenarios,
)


def probe(*, enforce: bool = True) -> Probe:
    """Create a probe. ``enforce=False`` skips the gate on exit (for exploration);
    you can still call ``result.gate()`` yourself."""
    return Probe(enforce=enforce)


__all__ = [
    "probe",
    "Probe",
    "ProbeResult",
    "ProbeError",
    "ProbeGateError",
    "ToolScenarios",
    "CallRecord",
    "ScenarioEvidence",
    "AssertionRecord",
    "GateReport",
    "SCHEMA_VERSION",
    # The built-in faults, for agents and tests that catch them
    "ToolCallError",
    "ToolCallTimeout",
    "ToolCallRateLimited",
    "ToolCallAPIError",
    "ToolCallMalformedOutput",
]
