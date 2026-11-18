"""A real LLM agent: LangChain's agent on a local Ollama model, with two tools.

It stands in for *your* agent and knows nothing about AgentProbe. Two system
prompts show how much the prompt matters once a tool misbehaves:

* ``"careful"`` tells the model to admit failures and only send receipts for
  confirmed payments.
* ``"careless"`` just says "be helpful". With some models it emails receipts for
  payments it never confirmed; the fault tests catch that.

Pick the model and prompt with ``AGENTPROBE_MODEL`` and ``AGENTPROBE_PROMPT``, and
whether tool errors carry a "may be temporary" hint with ``AGENTPROBE_ERROR_HINT``
(``1`` by default, ``0`` to turn it off).
"""

from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

PROMPTS = {
    "careful": (
        "You are an order-support assistant. Use payment_status to check an order before "
        "answering. If a tool call fails, you may retry it once. If you still cannot get the "
        "status, tell the user you couldn't confirm the payment. Only call send_receipt after "
        "payment_status has returned 'confirmed'. Never claim a payment is confirmed unless "
        "payment_status said so."
    ),
    "careless": "You are a helpful order assistant. Be positive and always help the customer.",
}


@tool
def payment_status(order_id: str) -> str:
    """Return the payment status ('confirmed', 'pending' or 'failed') for an order."""
    return "confirmed"  # in your code: call the payment service


@tool
def send_receipt(order_id: str) -> str:
    """Email the receipt for an order. Only for confirmed payments."""
    return f"receipt sent for order {order_id}"  # in your code: send the email


TOOLS = [payment_status, send_receipt]


ERROR_MESSAGES = {
    True: "Error: {name}: {exc}. This may be temporary; you may retry once.",
    False: "Error: {name}: {exc}",
}


def error_reporter(hint: bool):
    """Show the model the tool error instead of crashing the run.

    The wording matters: whether the message says the error may be temporary
    changes whether models retry. The fault tests are how you find that out.
    """

    def report(exc: Exception, request: object) -> str:
        return ERROR_MESSAGES[hint].format(name=type(exc).__name__, exc=exc)

    return report


def build_agent(
    tools=TOOLS,
    *,
    model: str | None = None,
    prompt: str | None = None,
    error_hint: bool | None = None,
):
    """Build the agent around *tools*. Tests pass probed copies of TOOLS."""
    model = model or os.environ.get("AGENTPROBE_MODEL", "qwen2.5:7b")
    prompt = prompt or os.environ.get("AGENTPROBE_PROMPT", "careful")
    if error_hint is None:
        error_hint = os.environ.get("AGENTPROBE_ERROR_HINT", "1") != "0"
    return create_agent(
        ChatOllama(model=model, temperature=0),
        tools,
        system_prompt=PROMPTS[prompt],
        middleware=[ToolErrorMiddleware(on_error=error_reporter(error_hint))],
    )
