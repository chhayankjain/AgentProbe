"""Run a third-party agent under AgentProbe without editing its code.

Target: the email assistant in langchain-ai/agents-from-scratch (commit 603fc7a).
That module builds a GPT-4.1 client and its tools when it is imported, and has no
parameter for passing tools in. So we import it with a placeholder OpenAI key and a
dead OpenAI address (no request can leave the machine), then replace three module
globals: the router model, the tool-calling model, and the name -> tool table.
Their prompts, nodes and edges run unchanged.

This "swap the module globals" pattern works for any agent that builds its tools at
import time. If you own the agent, prefer adding a ``tools`` parameter instead.
"""

from __future__ import annotations

import os
import sys

UPSTREAM = os.environ.get("AGENTS_FROM_SCRATCH_DIR", "")


def load():
    """Import the upstream email assistant. Returns (module, sample meeting email)."""
    if not UPSTREAM or not os.path.isdir(os.path.join(UPSTREAM, "src", "email_assistant")):
        raise RuntimeError("set AGENTS_FROM_SCRATCH_DIR to a checkout of agents-from-scratch")
    sys.path.insert(0, os.path.join(UPSTREAM, "src"))
    os.environ["OPENAI_API_KEY"] = "placeholder-not-used"
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"  # nothing listens here

    import email_assistant.email_assistant as ea
    from email_assistant.eval.email_dataset import email_input_4

    return ea, email_input_4


def use_local_model(ea, probe, model: str = "qwen2.5:7b"):
    """Point the assistant at a local Ollama model and probed copies of its tools."""
    from email_assistant.schemas import RouterSchema
    from langchain_ollama import ChatOllama

    wrapped = probe.wrap(ea.tools)
    llm = ChatOllama(model=model, temperature=0)
    ea.llm_router = llm.with_structured_output(RouterSchema)
    ea.llm_with_tools = llm.bind_tools(wrapped, tool_choice="any")
    ea.tools_by_name = {t.name: t for t in wrapped}
    return ea.email_assistant
