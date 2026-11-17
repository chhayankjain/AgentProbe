"""An ordinary tool-using agent: answers "did my order go through?".

It stands in for *your* agent. Nothing here knows about AgentProbe. The agent
takes its tools as arguments, which is the only thing AgentProbe needs: in the
tests we pass in probed copies of the same tools.

Two versions of the payment step show the bug AgentProbe is built to catch:

* ``answer(...)`` retries the payment check once, then admits it couldn't confirm.
* ``answer(..., broken=True)`` swallows the error and reports success anyway.
"""

from __future__ import annotations

import re
from collections.abc import Callable

Tools = dict[str, Callable[[str], str]]


# ---- the real tools (in your code these call your payment and email services) ----


def payment_status(order_id: str) -> str:
    """Return the payment status for an order."""
    return "confirmed"


def send_receipt(order_id: str) -> str:
    """Email the receipt for an order."""
    return f"receipt sent for order {order_id}"


TOOLS: Tools = {"payment_status": payment_status, "send_receipt": send_receipt}


# ---- the agent ----


def answer(question: str, tools: Tools = TOOLS, *, broken: bool = False) -> str:
    match = re.search(r"\d+", question)
    if not match:
        return "Which order do you mean?"
    order_id = match.group()

    if broken:
        # BUG: a failed payment check is treated as success.
        try:
            status = tools["payment_status"](order_id)
        except Exception:
            status = "confirmed"
    else:
        status = None
        for _ in range(2):  # one retry
            try:
                status = tools["payment_status"](order_id)
                break
            except Exception:
                continue
        if status is None:
            return (
                f"I couldn't confirm the payment for order {order_id} right now. "
                "Please try again in a few minutes."
            )

    if status != "confirmed":
        return f"Order {order_id} is not paid yet (status: {status})."
    receipt = tools["send_receipt"](order_id)
    return f"Order {order_id} is confirmed and paid; {receipt}."


if __name__ == "__main__":
    print(answer("Did order 42 go through?"))
