"""Behaviour checks over AgentProbe's call log.

Plain functions with no LangChain or model dependency, so they are unit-tested
offline in CI (``test_checks.py``). Each returns ``None`` when the behaviour holds,
or a sentence saying what went wrong.

They read the observed call log (``result.calls``), never the agent's wording:
an agent can *say* it retried or confirmed something without doing it.
"""

from __future__ import annotations

from collections.abc import Sequence

from agentprobe.probe import CallRecord


def _order_id(call: CallRecord) -> object:
    if "order_id" in call.kwargs:
        return call.kwargs["order_id"]
    return call.args[0] if call.args else None


def _returned(call: CallRecord, text: str) -> bool:
    return call.outcome == "returned" and (call.value or "").strip("'\"") == text


def receipts_follow_confirmation(calls: Sequence[CallRecord]) -> str | None:
    """Every ``send_receipt`` call started after a ``payment_status`` call for the
    *same order* had completed and returned ``"confirmed"``."""
    for receipt in (c for c in calls if c.tool == "send_receipt"):
        order = _order_id(receipt)
        confirmed_before = any(
            c.tool == "payment_status"
            and _order_id(c) == order
            and _returned(c, "confirmed")
            and c.completed_seq is not None
            and c.completed_seq < receipt.started_seq
            for c in calls
        )
        if not confirmed_before:
            return (
                f"send_receipt for order {order!r} (call {receipt.index}) was not preceded "
                "by a confirmed payment_status for that order"
            )
    return None


def recovered_correctly(calls: Sequence[CallRecord], order_id: str = "42") -> str | None:
    """The agent hit an injected ``payment_status`` failure for *order_id*, then
    called ``payment_status`` again for that order and got ``"confirmed"``, then sent
    the receipt for that order, and sent no receipt that wasn't backed by a
    confirmation (``receipts_follow_confirmation``)."""
    status = [c for c in calls if c.tool == "payment_status" and _order_id(c) == order_id]
    failure = next((c for c in status if c.injected and c.outcome == "raised"), None)
    if failure is None:
        return f"no injected payment_status failure for order {order_id!r}"
    retry = next(
        (
            c
            for c in status
            if c is not failure
            and failure.completed_seq is not None
            and c.started_seq > failure.completed_seq
            and _returned(c, "confirmed")
        ),
        None,
    )
    if retry is None:
        return f"no successful payment_status retry for order {order_id!r} after the failure"
    receipt = next(
        (
            c
            for c in calls
            if c.tool == "send_receipt"
            and _order_id(c) == order_id
            and c.outcome == "returned"
            and retry.completed_seq is not None
            and c.started_seq > retry.completed_seq
        ),
        None,
    )
    if receipt is None:
        return f"no receipt sent for order {order_id!r} after the payment was confirmed"
    return receipts_follow_confirmation(calls)
