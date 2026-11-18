"""Run a third-party agent under AgentProbe without editing its code.

Target: the email assistant in langchain-ai/agents-from-scratch. That module builds
a GPT-4.1 client and its tools when it is imported, and has no parameter for
passing tools in. ``upstream()`` imports it with a placeholder OpenAI key and a
dead OpenAI address (no request can leave the machine); ``local_agent()`` swaps
three module globals (the router model, the tool-calling model, and the
name -> tool table) for a local Ollama model and AgentProbe-wrapped tools. Their
prompts, nodes and edges run unchanged.

Both are context managers and undo everything on exit: environment variables,
``sys.path``, the imported upstream modules, and the swapped globals. So this
example can run alongside other tests without affecting them.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager

ENV = {"OPENAI_API_KEY": "placeholder-not-used", "OPENAI_BASE_URL": "http://127.0.0.1:9/v1"}


@contextmanager
def upstream(checkout: str):
    """Import the email assistant from *checkout*. Yields (module, sample email)."""
    src = os.path.join(os.path.abspath(checkout), "src")
    if not os.path.isdir(os.path.join(src, "email_assistant")):
        raise RuntimeError(f"{checkout} is not an agents-from-scratch checkout")
    saved_env = {k: os.environ.get(k) for k in ENV}
    saved_path = list(sys.path)
    before = set(sys.modules)
    os.environ.update(ENV)
    sys.path.insert(0, src)
    try:
        import email_assistant.email_assistant as ea
        from email_assistant.eval.email_dataset import email_input_4

        if not os.path.abspath(ea.__file__).startswith(src):
            raise RuntimeError(f"imported {ea.__file__}, expected the module under {src}")
        yield ea, email_input_4
    finally:
        for name in set(sys.modules) - before:
            if name == "email_assistant" or name.startswith("email_assistant."):
                del sys.modules[name]
        sys.path[:] = saved_path
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def local_agent(ea, probe, model: str = "qwen2.5:7b", *, make_llm=None, router_schema=None):
    """Point the assistant at a local model and probed copies of its tools.
    Yields the compiled graph; restores the three globals on exit.

    Every replacement is built before any global changes, so a failure while
    setting up (an unreachable model, a binding error) leaves the module as it was.
    ``make_llm`` and ``router_schema`` exist so this can be tested without a model.
    """
    if router_schema is None:
        from email_assistant.schemas import RouterSchema

        router_schema = RouterSchema
    if make_llm is None:
        from langchain_ollama import ChatOllama

        def make_llm():
            return ChatOllama(model=model, temperature=0)

    wrapped = probe.wrap(ea.tools)
    llm = make_llm()
    replacements = {
        "llm_router": llm.with_structured_output(router_schema),
        "llm_with_tools": llm.bind_tools(wrapped, tool_choice="any"),
        "tools_by_name": {t.name: t for t in wrapped},
    }
    saved = {name: getattr(ea, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(ea, name, value)
        yield ea.email_assistant
    finally:
        for name, value in saved.items():
            setattr(ea, name, value)
