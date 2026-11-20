"""Regex and predicate answer assertions follow the same rules as the others."""

from __future__ import annotations

import json
import re

import pytest

from agentprobe.testing import ProbeError, probe


def _result(answer=None, *, raises: Exception | None = None):
    def agent():
        if raises is not None:
            raise raises
        return answer

    with probe(enforce=False) as p:
        result = p.run(agent)
    return result


class TestMatches:
    def test_passes_when_the_pattern_is_found(self):
        r = _result("I couldn't verify order 42.")
        r.assert_answer_matches(r"could(n't| not) (confirm|verify)")
        assert r.assertions[-1].status == "pass"
        assert "couldn't verify" in r.assertions[-1].message

    def test_case_sensitive_unless_asked(self):
        r = _result("ORDER CONFIRMED")
        with pytest.raises(AssertionError):
            r.assert_answer_matches(r"confirmed")
        r.assert_answer_matches(r"confirmed", flags=re.IGNORECASE)
        r.assert_answer_matches(r"(?i)confirmed")
        r.assert_answer_matches(re.compile("confirmed", re.IGNORECASE))

    def test_fails_with_the_answer_in_the_message(self):
        r = _result("Order 42 is paid.")
        with pytest.raises(AssertionError, match="answer was 'Order 42 is paid.'"):
            r.assert_answer_matches(r"refund")

    def test_not_matches(self):
        r = _result("Order 42 is confirmed and paid.")
        with pytest.raises(AssertionError, match="matched 'confirmed'"):
            r.assert_answer_not_matches(r"confirm(ed)?")
        r.assert_answer_not_matches(r"refund")
        assert r.assertions[-1].status == "pass"

    def test_invalid_patterns_are_a_probe_error(self):
        r = _result("x")
        with pytest.raises(ProbeError, match="invalid regular expression"):
            r.assert_answer_matches("(unclosed")
        with pytest.raises(ProbeError, match="re.compile"):
            r.assert_answer_matches(re.compile("x"), flags=re.IGNORECASE)


class TestPredicate:
    def test_passes_on_a_truthy_result(self):
        r = _result('{"status": "pending", "order": 42}')
        r.assert_answer(lambda a: json.loads(a)["status"] == "pending", "status is pending")
        assert r.assertions[-1].name == "answer satisfies status is pending"
        assert r.assertions[-1].status == "pass"

    def test_names_the_check_after_the_function(self):
        def mentions_order_id(answer):
            return "42" in answer

        r = _result("Order 42")
        r.assert_answer(mentions_order_id)
        assert r.assertions[-1].name == "answer satisfies mentions_order_id"

    def test_fails_on_a_falsy_result(self):
        r = _result("Order 42 is paid.")
        with pytest.raises(AssertionError, match="answer was"):
            r.assert_answer(lambda a: "refund" in a, "mentions a refund")

    def test_a_check_that_raises_fails_and_says_why(self):
        r = _result("not json")
        with pytest.raises(AssertionError, match="the check raised JSONDecodeError"):
            r.assert_answer(json.loads, "answer is JSON")
        assert r.assertions[-1].status == "fail"


@pytest.mark.parametrize(
    "check",
    [
        lambda r: r.assert_answer_matches(r".*"),
        lambda r: r.assert_answer_not_matches(r"anything"),
        lambda r: r.assert_answer(lambda a: True, "always true"),
    ],
)
def test_no_answer_never_passes(check):
    """A crash or a None answer is not "an answer that matches" or "doesn't match"."""
    for r in (_result(None), _result(raises=RuntimeError("boom"))):
        with pytest.raises(AssertionError, match="no answer"):
            check(r)
        assert r.assertions[-1].status == "fail"


def test_a_required_failure_reaches_the_gate():
    with pytest.raises(AssertionError):
        with probe() as p:
            result = p.run(lambda: "Order 42 is confirmed.")
            try:
                result.assert_answer_not_matches(r"confirmed")
            except AssertionError:
                pass  # caught inside the block; the gate must still fail


class _AsyncCallable:
    async def __call__(self, answer: str) -> bool:
        return True


class _Awaitable:
    def __await__(self):
        yield
        return True


async def _async_check(answer: str) -> bool:
    return True  # would "pass" if the coroutine's truthiness were used


@pytest.mark.parametrize(
    "check",
    [
        _async_check,
        _AsyncCallable(),
        lambda answer: _async_check(answer),  # sync function returning a coroutine
        lambda answer: _Awaitable(),  # sync function returning another awaitable
    ],
    ids=["async-function", "async-callable-object", "returns-coroutine", "returns-awaitable"],
)
def test_awaitable_checks_fail_instead_of_passing_unrun(check, recwarn):
    r = _result("anything")
    with pytest.raises(AssertionError, match="returned an awaitable, so it never ran"):
        r.assert_answer(check, "async check")
    assert r.assertions[-1].status == "fail"
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_an_awaitable_check_fails_the_gate():
    with pytest.raises(AssertionError):
        with probe() as p:
            result = p.run(lambda: "Order 42 is paid.")
            try:
                result.assert_answer(_async_check)
            except AssertionError:
                pass  # caught inside the block; the gate must still fail


class _NoTruthValue:
    """Like a multi-element NumPy array: bool() raises."""

    def __bool__(self) -> bool:
        raise ValueError("The truth value of an array with more than one element is ambiguous")


def test_a_result_without_a_truth_value_fails_and_is_recorded():
    r = _result("anything")
    with pytest.raises(AssertionError, match="has no truth value \\(ValueError"):
        r.assert_answer(lambda a: _NoTruthValue(), "ambiguous result")
    assert r.assertions[-1].status == "fail"


def test_a_result_without_a_truth_value_fails_the_gate_when_caught():
    with pytest.raises(AssertionError):
        with probe() as p:
            result = p.run(lambda: "Order 42 is paid.")
            try:
                result.assert_answer(lambda a: _NoTruthValue(), "ambiguous result")
            except AssertionError:
                pass  # caught inside the block; the gate must still fail
    assert len(result.assertions) == 1 and result.assertions[0].status == "fail"
    assert not result.gate().passed
