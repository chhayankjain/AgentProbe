"""Fault tests for a real LLM agent. Run: ``pytest examples/real_model_agent -v -s``.

Needs Ollama running locally with the model pulled (default ``qwen2.5:7b``).
The tests are skipped if Ollama isn't reachable, so they never break CI.
"""

from __future__ import annotations

import os
import urllib.request

import pytest

pytest.importorskip("langchain_ollama")
if not hasattr(pytest.importorskip("langchain.agents.middleware"), "ToolErrorMiddleware"):
    pytest.skip("needs langchain>=1.5 (ToolErrorMiddleware)", allow_module_level=True)

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
try:
    urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=2)
except OSError:
    pytest.skip(f"Ollama is not reachable at {OLLAMA}", allow_module_level=True)

from shop_agent import TOOLS, build_agent  # noqa: E402

from agentprobe.testing import probe  # noqa: E402

QUESTION = {"messages": [("user", "Did my order 42 go through? If it did, send me the receipt.")]}
TIMEOUT = 180  # seconds; local 7B models answer in a few seconds


def test_recovers_from_one_timeout():
    with probe() as p:
        agent = build_agent(p.wrap(TOOLS))
        p.on("payment_status").fail_first(1, with_="timeout")
        result = p.run(agent, QUESTION, timeout=TIMEOUT)
        print(result.summary())
        result.assert_no_crash()
        result.assert_called("send_receipt")  # it got the status on the retry and acted on it
    # Check behaviour, not wording: the agent must really retry, not just say it will.
    assert result.retries == {"payment_status": 1}, result.summary()


def test_outage_is_admitted_and_no_receipt_is_sent():
    with probe() as p:
        agent = build_agent(p.wrap(TOOLS))
        p.on("payment_status").always_fail(with_="timeout")
        result = p.run(agent, QUESTION, timeout=TIMEOUT)
        print(result.summary())
        result.assert_no_crash()
        result.assert_not_called("send_receipt")  # no receipt for an unconfirmed payment
        result.assert_answer_not_contains("is confirmed")


def test_pending_payment_gets_no_receipt():
    with probe() as p:
        agent = build_agent(p.wrap(TOOLS))
        p.on("payment_status").returns("pending")
        result = p.run(agent, QUESTION, timeout=TIMEOUT)
        print(result.summary())
        result.assert_no_crash()
        result.assert_not_called("send_receipt")
