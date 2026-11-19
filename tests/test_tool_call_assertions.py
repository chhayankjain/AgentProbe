"""Tool-call assertions use the frozen evidence seen at invocation."""

import collections
import dataclasses
import datetime
import decimal
import enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pydantic
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


# ---- exact argument comparison (lossy report snapshots must never match) ----------


class _LongRepr:
    def __init__(self, tail: str) -> None:
        self.tail = tail

    def __repr__(self) -> str:
        return "LongRepr(" + "x" * 300 + self.tail + ")"


class _Color(enum.Enum):
    RED = "red"


@dataclasses.dataclass
class _Order:
    order_id: str
    items: list


@dataclasses.dataclass(eq=False)
class _IdentityOrder:
    order_id: str


class _Model(pydantic.BaseModel):
    order_id: str
    secret: str = pydantic.Field(exclude=True)


class _Timestamp(datetime.datetime):
    """A datetime subclass, like pandas' Timestamp, free to define its own equality."""


def _one_call(*args, **kwargs):
    with probe(enforce=False) as p:
        tool = p.wrap({"t": lambda *a, **k: None})["t"]
        result = p.run(lambda: tool(*args, **kwargs))
    return result


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (b"a" * 32 + b"X", b"a" * 32 + b"Y"),
        ({1: "a"}, {"1": "a"}),
        ((1, 2), [1, 2]),
        ([1, 2], (1, 2)),
        (datetime.date(2026, 1, 1), "2026-01-01"),
        (_Color.RED, "red"),
        (decimal.Decimal("1.5"), "1.5"),
        ({1, 2}, [1, 2]),
        ({"nested": {"k": b"a" * 40 + b"1"}}, {"nested": {"k": b"a" * 40 + b"2"}}),
    ],
)
def test_called_with_never_matches_a_value_that_only_looks_the_same(actual, expected):
    result = _one_call(actual)
    with pytest.raises(AssertionError):
        result.assert_called_with("t", expected)
    assert result.assertions[-1].status == "fail"


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (b"a" * 100, b"a" * 100),
        (b"abc", bytearray(b"abc")),
        ({3, 1, 2}, {2, 3, 1}),
        ({1: "a", "1": "b"}, {"1": "b", 1: "a"}),
        (_Color.RED, _Color.RED),
        (datetime.datetime(2026, 1, 1, 9), datetime.datetime(2026, 1, 1, 9)),
        (1, 1.0),
    ],
)
def test_called_with_matches_equal_values(actual, expected):
    _one_call(actual).assert_called_with("t", expected)


def test_called_with_compares_the_value_at_call_time():
    order = {"order_id": "42", "items": [1]}
    with probe(enforce=False) as p:
        tool = p.wrap({"t": lambda o: None})["t"]
        result = p.run(lambda: tool(order))
    order["items"].append(2)
    result.assert_called_with("t", {"order_id": "42", "items": [1]})
    with pytest.raises(AssertionError):
        result.assert_called_with("t", {"order_id": "42", "items": [1, 2]})


@pytest.mark.parametrize(
    "expected",
    [
        _LongRepr("A"),
        {"nested": object()},  # nested: must still raise ProbeError, not AttributeError
        [1, (2, {"k": _LongRepr("A")})],
        _Order("42", [1]),  # dataclasses can override equality
        _IdentityOrder("42"),
        _Model(order_id="42", secret="x"),  # model_dump() drops excluded fields
        collections.OrderedDict(a=1),  # order-sensitive equality
        _Timestamp(2026, 1, 1),  # subclasses can define their own equality
    ],
)
def test_called_with_rejects_an_expectation_it_cannot_compare(expected):
    result = _one_call("x")
    with pytest.raises(ProbeError, match="can't compare"):
        result.assert_called_with("t", expected)


@pytest.mark.parametrize(
    ("actual", "looks_like"),
    [
        (_Model(order_id="42", secret="x"), {"order_id": "42"}),
        (collections.OrderedDict(a=1, b=2), {"a": 1, "b": 2}),
        (_IdentityOrder("42"), {"order_id": "42"}),
    ],
)
def test_a_recorded_object_with_its_own_equality_is_never_matched(actual, looks_like):
    result = _one_call(actual)
    with pytest.raises(AssertionError):
        result.assert_called_with("t", looks_like)
    assert result.assertions[-1].status == "unassessed"


def test_an_argument_that_cannot_be_compared_is_unassessed_not_passed():
    with pytest.raises(ProbeGateError):
        with probe() as p:
            tool = p.wrap({"t": lambda o: None})["t"]
            result = p.run(lambda: tool(_LongRepr("A")))
            with pytest.raises(AssertionError):
                result.assert_called_with("t", "LongRepr")
            assert result.assertions[-1].status == "unassessed"
            assert "can't be compared exactly" in result.assertions[-1].message


def test_exact_copies_stay_out_of_reports():
    result = _one_call(b"a" * 64, _LongRepr("A"), key={1, 2})
    data = result.to_dict()
    json.dumps(data)
    call = data["observed"]["calls"][0]
    assert not {"exact_args", "exact_kwargs", "signature"} & set(call)


# ---- positional and keyword forms of the same argument ----------------------------


def payment_status(order_id: str, currency: str = "USD") -> str:
    return "paid"


def test_positional_and_keyword_forms_match_by_parameter_name():
    with probe(enforce=False) as p:
        tool = p.wrap(payment_status)
        result = p.run(lambda: (tool(order_id="42"), tool("43", "EUR")))
    result.assert_called_with("payment_status", "42")
    result.assert_called_with("payment_status", order_id="43", currency="EUR")
    result.assert_called_with("payment_status", "43", currency="EUR")
    with pytest.raises(AssertionError):
        result.assert_called_with("payment_status", "42", currency="EUR")


def test_langchain_tool_arguments_match_in_either_form():
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def lookup(order_id: str) -> str:
        """Look up an order."""
        return "paid"

    with probe(enforce=False) as p:
        (wrapped,) = p.wrap([lookup])
        result = p.run(lambda: wrapped.invoke({"order_id": "42"}))
    result.assert_called_with("lookup", "42")
    result.assert_called_with("lookup", order_id="42")
    with pytest.raises(AssertionError):
        result.assert_called_with("lookup", "43")


def test_tools_sharing_a_name_fall_back_to_the_arguments_as_passed():
    with probe(enforce=False) as p:
        first = p.wrap({"t": lambda order_id: None})["t"]
        p.wrap({"t": lambda other, extra: None})
        result = p.run(lambda: first(order_id="42"))
    result.assert_called_with("t", order_id="42")
    with pytest.raises(AssertionError):
        result.assert_called_with("t", "42")
