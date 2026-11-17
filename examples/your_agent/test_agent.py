"""Fault tests for the example agent. Run: ``pytest examples/your_agent``.

Each test wraps the agent's real tools, declares one fault, runs the agent, and
checks the outcome. When the ``with`` block ends, AgentProbe also checks that
every declared fault actually fired, so an agent that skips the tool can't pass.
"""

from __future__ import annotations

import pytest
from agent import TOOLS, answer

from agentprobe.testing import ProbeGateError, probe

QUESTION = "Did order 42 go through?"


def test_recovers_from_one_timeout():
    with probe() as p:
        tools = p.wrap(TOOLS)
        p.on("payment_status").fail_first(1, with_="timeout")
        result = p.run(lambda q: answer(q, tools), QUESTION)
        result.assert_no_crash()
        result.assert_answer_contains("confirmed")
    assert result.retries == {"payment_status": 1}


def test_admits_failure_when_payment_service_is_down():
    with probe() as p:
        tools = p.wrap(TOOLS)
        p.on("payment_status").always_fail(with_="timeout")
        result = p.run(lambda q: answer(q, tools), QUESTION)
        result.assert_no_crash()
        result.assert_answer_not_contains("is confirmed")  # never claim unverified success
        result.assert_answer_contains("couldn't confirm")
        result.assert_not_called("send_receipt")  # no receipt for an unconfirmed payment


def test_does_not_trust_a_missing_status():
    with probe() as p:
        tools = p.wrap(TOOLS)
        p.on("payment_status").returns("unknown")
        result = p.run(lambda q: answer(q, tools), QUESTION)
        result.assert_answer_contains("not paid yet")


# ---- the same test catches the deliberately broken agent ----


def test_broken_agent_is_caught():
    """The broken agent reports success without confirming payment. This is the
    test from above, run against ``broken=True``: it must fail."""
    with pytest.raises(AssertionError, match="does not contain 'is confirmed'"):
        with probe() as p:
            tools = p.wrap(TOOLS)
            p.on("payment_status").always_fail(with_="timeout")
            result = p.run(lambda q: answer(q, tools, broken=True), QUESTION)
            result.assert_answer_not_contains("is confirmed")


def test_skipping_the_tool_cannot_pass():
    """An agent that answers from a cache never meets the fault. AgentProbe
    reports the scenario as not exercised instead of letting the test pass."""

    def cached_agent(question: str) -> str:
        return "Order 42 is confirmed and paid."

    with pytest.raises(ProbeGateError, match="never called 'payment_status'"):
        with probe() as p:
            p.wrap(TOOLS)
            p.on("payment_status").always_fail(with_="timeout")
            result = p.run(cached_agent, QUESTION)
            result.assert_answer_contains("confirmed")
