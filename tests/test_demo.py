"""The no-model demo and scripted model must work offline, deterministically."""

from __future__ import annotations

import os
import runpy

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agentprobe.demo import FAULTS, main, run_demo
from agentprobe.fake_llm import ScriptedToolCallingModel


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression, {"__builtins__": {}}))  # noqa: S307


class TestScriptedModel:
    def test_calls_bound_tool_then_answers(self):
        llm = ScriptedToolCallingModel().bind_tools([calculator])
        first = llm.invoke([HumanMessage(content="Calculate 15% of 847.")])
        assert isinstance(first, AIMessage) and first.tool_calls
        call = first.tool_calls[0]
        assert call["name"] == "calculator" and call["args"] == {"expression": "15 * 847 / 100"}

        second = llm.invoke(
            [
                HumanMessage(content="q"),
                first,
                ToolMessage(content="127.05", tool_call_id=call["id"]),
            ]
        )
        assert second.content == "The result is 127.05."

    def test_error_result_yields_error_answer(self):
        llm = ScriptedToolCallingModel().bind_tools([calculator])
        first = llm.invoke([HumanMessage(content="q")])
        reply = llm.invoke(
            [
                HumanMessage(content="q"),
                first,
                ToolMessage(
                    content="Error: tool unavailable", tool_call_id=first.tool_calls[0]["id"]
                ),
            ]
        )
        assert reply.content.startswith("I could not complete the task")

    def test_falls_back_to_first_bound_tool(self):
        @tool
        def web_search(query: str) -> str:
            """Search."""
            return "x"

        llm = ScriptedToolCallingModel(tool_name="nope").bind_tools([web_search])
        assert llm.invoke([HumanMessage(content="q")]).tool_calls[0]["name"] == "web_search"


class TestDemo:
    @pytest.mark.parametrize("fault", sorted(FAULTS))
    def test_every_fault_type_runs_offline(self, fault: str):
        out = run_demo(fault=fault, probability=0.5, n_runs=6, seed=7, quiet=True)
        m, inj = out["metrics"], out["injector"]
        assert m.n_executed == 6 and m.n_backend_errors == 0
        assert inj["total_calls"] >= 6
        assert inj["injected_failures"] > 0  # p=0.5 over >= 6 calls with seed 7

    def test_seed_makes_it_reproducible(self):
        a = run_demo(probability=0.4, n_runs=8, seed=3, quiet=True)
        b = run_demo(probability=0.4, n_runs=8, seed=3, quiet=True)
        assert [r.success for r in a["results"]] == [r.success for r in b["results"]]
        assert a["injector"]["injected_failures"] == b["injector"]["injected_failures"]

    def test_baseline_has_no_failures(self):
        out = run_demo(probability=0.0, n_runs=5, quiet=True)
        assert out["metrics"].failure_rate == 0.0 and out["injector"]["injected_failures"] == 0

    def test_cli_entry_point(self, capsys: pytest.CaptureFixture[str]):
        assert main(["--fault", "timeout", "--p", "0.5", "--runs", "4"]) == 0
        captured = capsys.readouterr().out
        assert "Fault: timeout" in captured and "injection log" in captured

    def test_examples_quickstart_runs(self, capsys: pytest.CaptureFixture[str]):
        path = os.path.join(os.path.dirname(__file__), "..", "examples", "quickstart.py")
        runpy.run_path(path, run_name="__main__")
        assert "failure rate" in capsys.readouterr().out
