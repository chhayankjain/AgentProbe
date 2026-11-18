"""Offline tests for harness.py: no model, no upstream checkout. Run in CI.

They check that the harness leaves nothing behind, including when setup fails.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
from harness import ENV, local_agent, upstream

from agentprobe.testing import probe


class FakeLLM:
    def __init__(self, fail_on_bind=False):
        self.fail_on_bind = fail_on_bind

    def with_structured_output(self, schema):
        return ("router", schema)

    def bind_tools(self, tools, tool_choice=None):
        if self.fail_on_bind:
            raise RuntimeError("binding failed")
        return ("bound", len(tools))


def fake_module():
    from langchain_core.tools import tool

    @tool
    def lookup(day: str) -> str:
        """Look up availability (a stand-in for the upstream LangChain tools)."""
        return "free"

    return SimpleNamespace(
        tools=[lookup],
        llm_router="ORIGINAL_ROUTER",
        llm_with_tools="ORIGINAL_TOOLS_LLM",
        tools_by_name="ORIGINAL_TABLE",
        email_assistant="GRAPH",
    )


def originals(ea):
    return (ea.llm_router, ea.llm_with_tools, ea.tools_by_name)


def test_local_agent_swaps_and_restores():
    ea = fake_module()
    with probe(enforce=False) as p:
        with local_agent(ea, p, make_llm=FakeLLM, router_schema="Schema") as graph:
            assert graph == "GRAPH"
            assert ea.llm_router == ("router", "Schema")
            assert ea.llm_with_tools == ("bound", 1)
            assert set(ea.tools_by_name) == {"lookup"}
    assert originals(ea) == ("ORIGINAL_ROUTER", "ORIGINAL_TOOLS_LLM", "ORIGINAL_TABLE")


def test_local_agent_restores_when_the_body_raises():
    ea = fake_module()
    with probe(enforce=False) as p, pytest.raises(ValueError):
        with local_agent(ea, p, make_llm=FakeLLM, router_schema="Schema"):
            raise ValueError("test body failed")
    assert originals(ea) == ("ORIGINAL_ROUTER", "ORIGINAL_TOOLS_LLM", "ORIGINAL_TABLE")


def test_local_agent_changes_nothing_when_setup_fails():
    """bind_tools() fails after the router was built: no global may have changed."""
    ea = fake_module()
    with probe(enforce=False) as p, pytest.raises(RuntimeError, match="binding failed"):
        with local_agent(
            ea, p, make_llm=lambda: FakeLLM(fail_on_bind=True), router_schema="Schema"
        ):
            pass
    assert originals(ea) == ("ORIGINAL_ROUTER", "ORIGINAL_TOOLS_LLM", "ORIGINAL_TABLE")


def test_upstream_rejects_a_directory_that_is_not_a_checkout(tmp_path):
    before = (dict(os.environ), list(sys.path))
    with pytest.raises(RuntimeError, match="not an agents-from-scratch checkout"):
        with upstream(str(tmp_path)):
            pass
    assert (dict(os.environ), list(sys.path)) == before


def test_upstream_restores_everything_when_the_import_fails(tmp_path):
    pkg = tmp_path / "src" / "email_assistant"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "email_assistant.py").write_text("raise ImportError('broken upstream module')\n")
    env_before = {k: os.environ.get(k) for k in ENV}
    path_before, modules_before = list(sys.path), set(sys.modules)
    with pytest.raises(ImportError, match="broken upstream module"):
        with upstream(str(tmp_path)):
            pass
    assert {k: os.environ.get(k) for k in ENV} == env_before
    assert sys.path == path_before
    assert not {m for m in set(sys.modules) - modules_before if m.startswith("email_assistant")}
