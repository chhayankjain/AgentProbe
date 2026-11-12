"""Regression tests for the qualification experiment checker.

Validates that the checker correctly distinguishes:
  - wrong-tool / right-answer → FAIL
  - right-tool / wrong-arguments → FAIL
  - right-tool / wrong-operation-sharing-operand → FAIL
  - right-tool / right-arguments / wrong-return → FAIL
  - right-tool / right-arguments / wrong-answer → FAIL
  - right-tool / right-arguments / right-answer → PASS
  - fault at wrong tool → FAIL (guaranteed_fault phase)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from run_qualification import (
    BASELINE_CONFIG,
    GUARANTEED_FAULT_CONFIG,
    QUALIFICATION_TASKS,
    QualificationTrial,
    run_trial,
)


def _make_mock_agent(tool_name: str, call_args: tuple, return_value: str | None,
                     injected: bool = False, output: str = "12000",
                     exception_type: type | None = None):
    """Build a mock agent whose _injector log has exactly one record."""
    from agentprobe.injector.tool_failure import (
        InjectionResult, ToolFailureType, ToolFailureInjector, ToolFailureConfig,
    )

    agent = MagicMock()
    agent.invoke.return_value = {"output": output}

    exc = exception_type("injected") if exception_type else None
    rec = InjectionResult(
        tool_name=tool_name,
        failure_type=ToolFailureType.API_ERROR if injected else ToolFailureType.NONE,
        injected=injected,
        exception=exc,
        latency_ms=1.0,
        span_id="deadbeef00000001",
        call_index=1,
        call_args=call_args,
        call_kwargs={},
        return_value=return_value,
    )

    injector = MagicMock(spec=ToolFailureInjector)
    injector.injection_log = [rec]
    agent._injector = injector
    return agent


TASK_ADD = QUALIFICATION_TASKS[1]  # qual_add: 3571 + 8429 = 12000


class TestBaselinePhase:
    """Baseline qualification requires: correct answer, required tool called,
    full expression in arguments, and correct return value."""

    @patch("run_qualification.build_agent")
    def test_pass_all_checks(self, mock_build):
        mock_build.return_value = _make_mock_agent(
            tool_name="calculator",
            call_args=("3571 + 8429",),
            return_value="12000",
            output="The answer is 12000.",
        )
        trial = run_trial("langgraph", TASK_ADD, "baseline", "llama3.1", BASELINE_CONFIG)
        assert trial.verdict == "PASS", trial.failure_reasons

    @patch("run_qualification.build_agent")
    def test_wrong_tool_right_answer(self, mock_build):
        """web_search called instead of calculator, but output has correct answer."""
        mock_build.return_value = _make_mock_agent(
            tool_name="web_search",
            call_args=("3571 + 8429",),
            return_value="some search result",
            output="The answer is 12000.",
        )
        trial = run_trial("langgraph", TASK_ADD, "baseline", "llama3.1", BASELINE_CONFIG)
        assert trial.verdict == "FAIL"
        assert any("calculator" in r and "not called" in r for r in trial.failure_reasons)

    @patch("run_qualification.build_agent")
    def test_right_tool_wrong_args(self, mock_build):
        """Calculator called, but with completely unrelated expression."""
        mock_build.return_value = _make_mock_agent(
            tool_name="calculator",
            call_args=("999 + 1",),
            return_value="1000",
            output="The answer is 12000.",
        )
        trial = run_trial("langgraph", TASK_ADD, "baseline", "llama3.1", BASELINE_CONFIG)
        assert trial.verdict == "FAIL"
        assert any("3571 + 8429" in r and "not found" in r for r in trial.failure_reasons)

    @patch("run_qualification.build_agent")
    def test_wrong_operation_sharing_operand(self, mock_build):
        """Calculator called with '3571 - 1' (shares operand '3571' but wrong
        operation). Must FAIL — this is the counterexample that caught the
        old fragment-only predicate."""
        mock_build.return_value = _make_mock_agent(
            tool_name="calculator",
            call_args=("3571 - 1",),
            return_value="3570",
            output="The answer is 12000.",
        )
        trial = run_trial("langgraph", TASK_ADD, "baseline", "llama3.1", BASELINE_CONFIG)
        assert trial.verdict == "FAIL"
        assert any("3571 + 8429" in r and "not found" in r for r in trial.failure_reasons)

    @patch("run_qualification.build_agent")
    def test_correct_expression_wrong_return(self, mock_build):
        """Calculator called with correct expression but returns wrong value."""
        mock_build.return_value = _make_mock_agent(
            tool_name="calculator",
            call_args=("3571 + 8429",),
            return_value="11999",
            output="The answer is 12000.",
        )
        trial = run_trial("langgraph", TASK_ADD, "baseline", "llama3.1", BASELINE_CONFIG)
        assert trial.verdict == "FAIL"
        assert any("return value" in r.lower() and "11999" in r for r in trial.failure_reasons)

    @patch("run_qualification.build_agent")
    def test_right_tool_right_args_wrong_answer(self, mock_build):
        """Calculator called correctly, but output has wrong answer."""
        mock_build.return_value = _make_mock_agent(
            tool_name="calculator",
            call_args=("3571 + 8429",),
            return_value="12000",
            output="The answer is 36900.",
        )
        trial = run_trial("langgraph", TASK_ADD, "baseline", "llama3.1", BASELINE_CONFIG)
        assert trial.verdict == "FAIL"
        assert any("12000" in r and "not in output" in r for r in trial.failure_reasons)

    @patch("run_qualification.build_agent")
    def test_no_tool_calls(self, mock_build):
        """No tools called at all."""
        agent = MagicMock()
        agent.invoke.return_value = {"output": "The answer is 12000."}
        injector = MagicMock()
        injector.injection_log = []
        agent._injector = injector
        mock_build.return_value = agent

        trial = run_trial("langgraph", TASK_ADD, "baseline", "llama3.1", BASELINE_CONFIG)
        assert trial.verdict == "FAIL"
        assert any("not called" in r for r in trial.failure_reasons)


class TestGuaranteedFaultPhase:
    """Guaranteed-fault requires: fault injected, tool call recorded,
    fault at the required tool boundary."""

    @patch("run_qualification.build_agent")
    def test_pass_fault_at_required_tool(self, mock_build):
        from agentprobe.injector.tool_failure import ToolCallAPIError
        mock_build.return_value = _make_mock_agent(
            tool_name="calculator",
            call_args=("3571 + 8429",),
            return_value=None,
            injected=True,
            output="EXCEPTION: ToolCallAPIError",
            exception_type=ToolCallAPIError,
        )
        trial = run_trial("langgraph", TASK_ADD, "guaranteed_fault", "llama3.1",
                          GUARANTEED_FAULT_CONFIG)
        assert trial.verdict == "PASS", trial.failure_reasons

    @patch("run_qualification.build_agent")
    def test_fault_at_wrong_tool(self, mock_build):
        """Fault injected, but only at web_search, not calculator."""
        from agentprobe.injector.tool_failure import ToolCallAPIError
        mock_build.return_value = _make_mock_agent(
            tool_name="web_search",
            call_args=("query",),
            return_value=None,
            injected=True,
            output="EXCEPTION: ToolCallAPIError",
            exception_type=ToolCallAPIError,
        )
        trial = run_trial("langgraph", TASK_ADD, "guaranteed_fault", "llama3.1",
                          GUARANTEED_FAULT_CONFIG)
        assert trial.verdict == "FAIL"
        assert any("calculator" in r and "not injected" in r.lower()
                    for r in trial.failure_reasons)
