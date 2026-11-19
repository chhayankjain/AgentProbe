"""Tool-call assertions use the frozen evidence seen at invocation."""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from agentprobe.testing import ProbeError, ProbeGateError, probe


def test_called_with_matches_one_frozen_call():
    with probe(enforce=False) as p:
        lookup = p.wrap({"lookup": lambda *args, **kwargs: None})["lookup"]
        payload = {"ids": [42]}
        lookup("first", tool="other")
        lookup(payload, tool="payment", required=False, extra=1)
        result = p.run(lambda: None)
        payload["ids"].clear()

    result.assert_called_with("lookup", {"ids": [42]}, tool="payment", required=False)
    for args, kwargs in [
        ((payload,), {}),
        (("first",), {"tool": "payment"}),
        (({"ids": [42]},), {"tool": "payment", "required": True}),
        (({"ids": [42]},), {"absent": None}),
        ((), {"tool": "payment"}),
    ]:
        with pytest.raises(AssertionError):
            result.assert_called_with("lookup", *args, **kwargs)
        assert result.assertions[-1].status == "fail"
        assert "lookup(" in result.assertions[-1].message


def test_call_order_uses_invocation_and_consumes_each_occurrence():
    with probe(enforce=False) as p:
        inner = p.wrap({"check": lambda: None})["check"]
        outer = p.wrap({"submit": inner})["submit"]
        audit = p.wrap({"audit": lambda: None})["audit"]
        outer()
        audit()
        result = p.run(inner)

    # The nested check finishes before submit. It was invoked second.
    assert result.calls[1].completed_seq < result.calls[0].completed_seq
    result.assert_call_order(["submit", "check", "check"])
    result.assert_call_order([])
    with pytest.raises(ProbeError):
        result.assert_call_order("check")
    result.assert_call_order(["check", "submit"], required=False)
    assert result.assertions[-1].status == "fail"
    assert "submit(" in result.assertions[-1].message
    result.assert_call_order(["check"] * 3, required=False)
    assert result.assertions[-1].status == "fail"
    assert result.gate().passed


def test_zero_counts_and_a_required_failure_reach_the_gate():
    with pytest.raises(ProbeGateError, match="lookup"):
        with probe() as p:
            lookup = p.wrap({"lookup": lambda: None})["lookup"]
            result = p.run(lookup)
            result.assert_called_times("missing", 0).assert_called_at_most("missing", 0)
            result.assert_called_times("lookup", 1).assert_called_at_most("lookup", 2)
            result.assert_called_times("lookup", 2, required=False)
            assert result.assertions[-1].status == "fail"
            with pytest.raises(AssertionError):
                result.assert_called_at_most("lookup", 0)
            assert "lookup(" in result.gate().failures[0]


def test_pending_and_late_calls_at_a_deadline():
    entered = threading.Event()
    release = threading.Event()

    def lookup(order):
        entered.set()
        release.wait()

    with probe(enforce=False) as p, ThreadPoolExecutor(max_workers=1) as pool:
        wrapped = p.wrap(lookup)
        pending = pool.submit(wrapped, "original")
        try:
            assert entered.wait(5)
            result = p.run(pending.result, timeout=0)
        finally:
            release.set()
        pending.result(timeout=5)
        wrapped("late")

    assert result.still_running and result.calls[0].status == "pending"
    assert len(p.late_calls) == 1 and p.late_calls[0].after_probe_end
    # Include the late record explicitly to verify it cannot affect any verdict.
    result = replace(result, calls=[*result.calls, *p.late_calls])
    result.assert_called_with("lookup", "original").assert_call_order(["lookup"])
    with pytest.raises(AssertionError):
        result.assert_called_with("lookup", "late")
    assert result.assertions[-1].status == "unassessed"
    result.assert_call_order(["lookup", "lookup"], required=False)
    assert result.assertions[-1].status == "unassessed"
    for assertion in (result.assert_called_times, result.assert_called_at_most):
        for limit, status in [(0, "fail"), (1, "unassessed"), (2, "unassessed")]:
            assertion("lookup", limit, required=False)
            assert result.assertions[-1].status == status
            assert "late" not in result.assertions[-1].message
    assert not result.gate().passed


@pytest.mark.parametrize("invalid", [True, -1, 1.5])
def test_counts_require_nonnegative_integers(invalid):
    with probe() as p:
        result = p.run(lambda: None)
        for assertion in (result.assert_called_times, result.assert_called_at_most):
            with pytest.raises(ProbeError):
                assertion("lookup", invalid)
