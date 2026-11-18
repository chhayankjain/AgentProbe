"""Live fault tests for a real LLM agent. Run: ``pytest examples/real_model_agent -v -s``.

Needs Ollama running locally with the model pulled (default ``qwen2.5:7b``) and
``langchain>=1.5``. These tests are marked ``integration`` and skipped when Ollama
isn't reachable; CI runs only the offline checks in ``test_checks.py``.
"""

from __future__ import annotations

import os
import urllib.request

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("langchain_ollama")
if not hasattr(pytest.importorskip("langchain.agents.middleware"), "ToolErrorMiddleware"):
    pytest.skip("needs langchain>=1.5 (ToolErrorMiddleware)", allow_module_level=True)

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
try:
    urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=2)
except OSError:
    pytest.skip(f"Ollama is not reachable at {OLLAMA}", allow_module_level=True)

from checks import recovered_correctly  # noqa: E402
from shop_agent import TOOLS, build_agent  # noqa: E402

from agentprobe.testing import probe  # noqa: E402

QUESTION = {"messages": [("user", "Did my order 42 go through? If it did, send me the receipt.")]}
TIMEOUT = 180  # seconds; local 7B models answer in a few seconds


def test_recovers_from_one_timeout():
    """After one timeout the agent really re-checks order 42, gets "confirmed",
    and only then sends the receipt for order 42 (checked on the call log)."""
    with probe() as p:
        agent = build_agent(p.wrap(TOOLS))
        p.on("payment_status").fail_first(1, with_="timeout")
        result = p.run(agent, QUESTION, timeout=TIMEOUT)
        print(result.summary())
        result.assert_no_crash()
        problem = recovered_correctly(result.calls, order_id="42")
        assert problem is None, f"{problem}\n\n{result.summary()}"


def test_outage_sends_no_receipt():
    """While the payment service is down, the agent doesn't crash and sends no
    receipt. This checks tool behaviour only: whether the *answer* admits the
    failure needs an answer contract (for example structured output), which a
    substring check can't establish."""
    with probe() as p:
        agent = build_agent(p.wrap(TOOLS))
        p.on("payment_status").always_fail(with_="timeout")
        result = p.run(agent, QUESTION, timeout=TIMEOUT)
        print(result.summary())
        result.assert_no_crash()
        result.assert_not_called("send_receipt")


def test_pending_payment_gets_no_receipt():
    """The payment stays pending on every check, so no receipt may be sent."""
    with probe() as p:
        agent = build_agent(p.wrap(TOOLS))
        p.on("payment_status").returns("pending", times=None)
        result = p.run(agent, QUESTION, timeout=TIMEOUT)
        print(result.summary())
        result.assert_no_crash()
        result.assert_not_called("send_receipt")
