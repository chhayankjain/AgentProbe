"""Offline tests for checks.py (no model, run in CI). Each case is a call log
the checks must accept or reject, including the false passes found in review."""

from __future__ import annotations

from checks import receipts_follow_confirmation, recovered_correctly

from agentprobe.probe import CallRecord

_seq = 0


def call(tool, order, *, value=None, raised=False, injected=False):
    """A completed call; calls are created in time order and never overlap."""
    global _seq
    _seq += 2
    return CallRecord(
        index=_seq // 2,
        tool=tool,
        tool_call_index=1,
        args=[],
        kwargs={"order_id": order},
        status="completed",
        outcome="raised" if raised else "returned",
        value=None if raised else repr(value),
        injected=injected,
        started_seq=_seq - 1,
        completed_seq=_seq,
    )


def timeout(order):
    return call("payment_status", order, raised=True, injected=True)


def status(order, value):
    return call("payment_status", order, value=value)


def receipt(order):
    return call("send_receipt", order, value=f"receipt sent for order {order}")


def test_correct_recovery_passes():
    log = [timeout("42"), status("42", "confirmed"), receipt("42")]
    assert recovered_correctly(log) is None


def test_receipt_before_the_check_fails():
    """Review case: receipt first, the check for 42 fails, then order 99 is checked."""
    log = [receipt("42"), timeout("42"), status("99", "confirmed")]
    assert recovered_correctly(log) is not None


def test_retry_for_the_wrong_order_fails():
    log = [timeout("42"), status("99", "confirmed"), receipt("42")]
    assert "no successful payment_status retry" in recovered_correctly(log)


def test_receipt_for_another_order_fails():
    log = [timeout("42"), status("42", "confirmed"), receipt("42"), receipt("99")]
    assert "order '99'" in recovered_correctly(log)


def test_no_receipt_fails():
    log = [timeout("42"), status("42", "confirmed")]
    assert "no receipt sent" in recovered_correctly(log)


def test_pending_status_is_not_a_confirmation():
    log = [timeout("42"), status("42", "pending"), receipt("42")]
    assert recovered_correctly(log) is not None
    assert receipts_follow_confirmation([status("42", "pending"), receipt("42")]) is not None


def test_no_fault_means_no_recovery_to_judge():
    log = [status("42", "confirmed"), receipt("42")]
    assert "no injected payment_status failure" in recovered_correctly(log)
