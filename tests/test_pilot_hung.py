"""Regression test: pilot must reject hung invocations.

Verifies that run_pilot() returns False when a baseline or injection
condition leaves a lingering worker thread (invocation_hung=True),
even if the first trial succeeded.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from experiments.run_experiment_v2 import (
    ConditionOutcome,
    EnhancedRunRecord,
    run_pilot,
)


def _make_record(
    run_id: int,
    success: bool = True,
    grade_correct: bool = True,
    output: str = "Paris",
    status: str = "completed",
    failure_type: str = "none",
    injector_calls: int = 0,
    injector_injections: int = 0,
) -> EnhancedRunRecord:
    return EnhancedRunRecord(
        run_id=run_id,
        framework="langgraph",
        task="tool_use",
        failure_type=failure_type,
        batch_id="pilot",
        status=status,
        success=success,
        latency_ms=100.0,
        output=output,
        grade_correct=grade_correct,
        grade_score=1.0 if grade_correct else 0.0,
        injector_calls=injector_calls,
        injector_injections=injector_injections,
        prompt_index=0,
        input_text="What is the capital of France?",
    )


def _good_baseline() -> ConditionOutcome:
    """A normal baseline outcome: 2 successful runs, not hung."""
    return ConditionOutcome(
        records=[
            _make_record(0, success=True, grade_correct=True, output="Paris"),
            _make_record(1, success=True, grade_correct=True, output="Paris"),
        ],
        invocation_hung=False,
    )


def _good_injection() -> ConditionOutcome:
    """A normal injection outcome with tool calls and injections."""
    return ConditionOutcome(
        records=[
            _make_record(
                0,
                success=True,
                grade_correct=True,
                output="Paris",
                failure_type="api_error",
                injector_calls=5,
                injector_injections=2,
            ),
            _make_record(
                1,
                success=False,
                grade_correct=False,
                output="Error",
                failure_type="api_error",
                injector_calls=5,
                injector_injections=2,
            ),
        ],
        invocation_hung=False,
    )


def _hung_baseline() -> ConditionOutcome:
    """Baseline where run 0 succeeded but run 1 timed out and hung."""
    return ConditionOutcome(
        records=[
            _make_record(0, success=True, grade_correct=True, output="Paris"),
            _make_record(1, success=False, grade_correct=False, output="", status="completed"),
        ],
        invocation_hung=True,
    )


def _hung_injection() -> ConditionOutcome:
    """Injection where a trial hung."""
    return ConditionOutcome(
        records=[
            _make_record(
                0,
                success=True,
                grade_correct=True,
                output="Paris",
                failure_type="api_error",
                injector_calls=5,
                injector_injections=2,
            ),
            _make_record(
                1,
                success=False,
                grade_correct=False,
                output="",
                failure_type="api_error",
                injector_calls=5,
                injector_injections=2,
            ),
        ],
        invocation_hung=True,
    )


@patch("experiments.run_experiment_v2.run_single_condition")
def test_pilot_rejects_hung_baseline(mock_run):
    """Pilot must abort immediately on first hung baseline — only 1 condition called."""
    mock_run.return_value = _hung_baseline()

    with pytest.MonkeyPatch.context() as mp:
        mp.chdir("/tmp")
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path

            result = run_pilot("llama3.1", Path(tmpdir), seed=42)

    assert result is False, "Pilot should reject hung baseline invocation"
    # Immediate abort: only the first baseline should have been called
    assert mock_run.call_count == 1, (
        f"Pilot should abort after first hung baseline, but called "
        f"run_single_condition {mock_run.call_count} times"
    )


@patch("experiments.run_experiment_v2.run_single_condition")
def test_pilot_rejects_hung_injection(mock_run):
    """Pilot must abort immediately when injection hangs — only 2 conditions called."""
    mock_run.side_effect = [
        _good_baseline(),  # langgraph baseline — OK
        _hung_injection(),  # langgraph injection — hung → immediate abort
    ]

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_pilot("llama3.1", Path(tmpdir), seed=42)

    assert result is False, "Pilot should reject hung injection invocation"
    # Immediate abort: baseline + injection for first framework only
    assert mock_run.call_count == 2, (
        f"Pilot should abort after hung injection, but called "
        f"run_single_condition {mock_run.call_count} times"
    )


@patch("experiments.run_experiment_v2.run_single_condition")
def test_pilot_passes_when_nothing_hung(mock_run):
    """Pilot should pass when all conditions complete without hanging."""
    # 3 frameworks × (baseline + injection) = 6 calls
    mock_run.side_effect = [
        _good_baseline(),
        _good_injection(),  # langgraph
        _good_baseline(),
        _good_injection(),  # langchain
        _good_baseline(),
        _good_injection(),  # autogen
    ]

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_pilot("llama3.1", Path(tmpdir), seed=42)

    assert result is True, "Pilot should pass when nothing is hung"
