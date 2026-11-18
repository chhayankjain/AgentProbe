"""Fault tests for a third-party agent: the agents-from-scratch email assistant.

Run (see README.md for setup):

    AGENTS_FROM_SCRATCH_DIR=../agents-from-scratch pytest examples/third_party_agent -v -s

The tests assert the behaviour we'd *want*. Two of them fail today because tool
exceptions escape the assistant's tool node (reported upstream as
langchain-ai/agents-from-scratch#70), so they are marked ``xfail(strict=True)``:
they document the bug now, and turn red (XPASS) once it is fixed, so this example
gets updated.
"""

from __future__ import annotations

import os
import urllib.request

import pytest

pytestmark = pytest.mark.integration

if not os.environ.get("AGENTS_FROM_SCRATCH_DIR"):
    pytest.skip("set AGENTS_FROM_SCRATCH_DIR to run this example", allow_module_level=True)

from harness import load, use_local_model  # noqa: E402

from agentprobe.testing import probe  # noqa: E402

ea, MEETING_REQUEST = load()
UPSTREAM_BUG = "tool exceptions escape tool_node: langchain-ai/agents-from-scratch#70"


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen(os.environ.get("OLLAMA_HOST", "http://localhost:11434"), timeout=2)
    except OSError:
        return False
    return True


needs_ollama = pytest.mark.skipif(not _ollama_up(), reason="Ollama is not running")


# ---- no model needed --------------------------------------------------------


@pytest.mark.xfail(strict=True, raises=TimeoutError, reason=UPSTREAM_BUG)
def test_tool_node_reports_tool_errors_to_the_model():
    """A failing tool should come back to the model as an error message."""
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

    @tool
    def check_calendar_availability(day: str) -> str:
        """Check calendar availability for a given day."""
        raise TimeoutError("calendar service timed out")

    original = ea.tools_by_name
    ea.tools_by_name = {**original, "check_calendar_availability": check_calendar_availability}
    try:
        call = {"name": "check_calendar_availability", "args": {"day": "Tuesday"}, "id": "c1"}
        out = ea.tool_node({"messages": [AIMessage(content="", tool_calls=[call])]})
    finally:
        ea.tools_by_name = original
    assert out["messages"][0]["tool_call_id"] == "c1"


# ---- the real graph on a local model ----------------------------------------

INPUT = {"email_input": MEETING_REQUEST}


@needs_ollama
@pytest.mark.xfail(strict=True, reason=UPSTREAM_BUG)
def test_survives_one_calendar_timeout():
    with probe() as p:
        graph = use_local_model(ea, p)
        p.on("check_calendar_availability").fail_first(1, with_="timeout")
        result = p.run(graph, INPUT, timeout=300)
        print(result.summary())
        result.assert_no_crash()


@needs_ollama
@pytest.mark.xfail(strict=True, reason=UPSTREAM_BUG)
def test_survives_a_scheduling_outage():
    with probe() as p:
        graph = use_local_model(ea, p)
        p.on("schedule_meeting").always_fail(with_="api_error")
        result = p.run(graph, INPUT, timeout=300)
        print(result.summary())
        result.assert_no_crash()


@needs_ollama
def test_failed_scheduling_still_gets_a_reply():
    """When scheduling returns an error message (as the Gmail tool does), the agent
    should still reply instead of crashing or going silent."""
    failure = "Could not schedule the meeting: the calendar service rejected the request."
    with probe() as p:
        graph = use_local_model(ea, p)
        p.on("schedule_meeting").returns(failure, times=None)
        result = p.run(graph, INPUT, timeout=300)
        print(result.summary())
        result.assert_no_crash()
        result.assert_called("write_email")
