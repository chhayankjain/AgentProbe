"""Fault tests for a third-party agent: the agents-from-scratch email assistant.

Run (see README.md for setup):

    AGENTS_FROM_SCRATCH_DIR=../agents-from-scratch pytest examples/third_party_agent -v -s

The tests assert the behaviour we'd *want*: a failing tool shouldn't end the run.
On commit 603fc7a tool exceptions escape the assistant's tool node
(langchain-ai/agents-from-scratch#70). A test reports that known bug as an
expected failure only when the injected fault actually fired and produced exactly
that crash. Anything else (no model, a fault that never fired, a different
exception) is a real failure. With the proposed fix
(langchain-ai/agents-from-scratch#71) the model-free and timeout tests pass; the
outage test also needs open PR #65 (a separate bug, #61). See
results/validation-2026-11-18.md. "Pass" here means the run completes without an
unhandled exception; it doesn't check what the agent did next.

Faults use the exception types the fix handles (``TimeoutError``,
``ToolException``), because AgentProbe's own fault classes don't subclass them.
"""

from __future__ import annotations

import os
import urllib.request

import pytest

pytestmark = pytest.mark.integration

CHECKOUT = os.environ.get("AGENTS_FROM_SCRATCH_DIR")
if not CHECKOUT:
    pytest.skip("set AGENTS_FROM_SCRATCH_DIR to run this example", allow_module_level=True)

from harness import local_agent, upstream  # noqa: E402
from langchain_core.tools import ToolException  # noqa: E402

from agentprobe.testing import probe  # noqa: E402

KNOWN_BUG = "known upstream bug: tool exceptions escape tool_node (agents-from-scratch#70)"


@pytest.fixture(scope="module")
def assistant():
    with upstream(CHECKOUT) as loaded:
        yield loaded


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen(os.environ.get("OLLAMA_HOST", "http://localhost:11434"), timeout=2)
    except OSError:
        return False
    return True


needs_ollama = pytest.mark.skipif(not _ollama_up(), reason="Ollama is not running")


def expect_known_bug(result, injected: type[BaseException]) -> None:
    """xfail only for the documented crash: the fault fired and its exception escaped."""
    unexercised = [s.description for s in result.scenarios if s.status != "exercised"]
    assert not unexercised, f"fault never fired: {unexercised}\n\n{result.summary()}"
    if result.exception is not None and type(result.exception) is injected:
        pytest.xfail(KNOWN_BUG)


# ---- no model needed --------------------------------------------------------


def test_tool_node_reports_tool_errors_to_the_model(assistant):
    """A failing tool should come back to the model as an error message."""
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

    ea, _ = assistant

    @tool
    def check_calendar_availability(day: str) -> str:
        """Check calendar availability for a given day."""
        raise TimeoutError("calendar service timed out")

    original = ea.tools_by_name
    ea.tools_by_name = {**original, "check_calendar_availability": check_calendar_availability}
    call = {"name": "check_calendar_availability", "args": {"day": "Tuesday"}, "id": "c1"}
    try:
        out = ea.tool_node({"messages": [AIMessage(content="", tool_calls=[call])]})
    except TimeoutError:
        pytest.xfail(KNOWN_BUG)
    finally:
        ea.tools_by_name = original
    assert out["messages"][0]["tool_call_id"] == "c1"


# ---- the real graph on a local model ----------------------------------------


@needs_ollama
def test_survives_one_calendar_timeout(assistant):
    ea, email = assistant
    with probe() as p, local_agent(ea, p) as graph:
        p.on("check_calendar_availability").fail_first(
            1, with_=TimeoutError("calendar service timed out")
        )
        result = p.run(graph, {"email_input": email}, timeout=300)
        print(result.summary())
        expect_known_bug(result, TimeoutError)
        result.assert_no_crash()


@needs_ollama
def test_survives_a_scheduling_outage(assistant):
    ea, email = assistant
    with probe() as p, local_agent(ea, p) as graph:
        p.on("schedule_meeting").always_fail(with_=ToolException("calendar API returned 503"))
        result = p.run(graph, {"email_input": email}, timeout=300)
        print(result.summary())
        expect_known_bug(result, ToolException)
        result.assert_no_crash()


@needs_ollama
def test_failed_scheduling_still_gets_a_reply(assistant):
    """When scheduling returns an error message (as the Gmail tool does), the agent
    should still reply instead of crashing or going silent."""
    ea, email = assistant
    failure = "Could not schedule the meeting: the calendar service rejected the request."
    with probe() as p, local_agent(ea, p) as graph:
        p.on("schedule_meeting").returns(failure, times=None)
        result = p.run(graph, {"email_input": email}, timeout=300)
        print(result.summary())
        result.assert_no_crash()
        result.assert_called("write_email")
