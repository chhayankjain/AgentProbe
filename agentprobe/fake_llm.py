"""A deterministic, network-free stand-in for a tool-calling chat model.

Use it to try AgentProbe without Ollama or an API key, to run the adapters in
CI, and to write tests for your own tools and injectors.

The model follows a fixed script that mirrors a one-step ReAct loop:

1. On the first turn it calls one bound tool (``tool_name``, default
   ``calculator``) with ``tool_args``.
2. After the tool result comes back as a ``ToolMessage`` it answers in plain
   text, either ``answer_template`` or ``error_template`` depending on whether
   the tool result looks like an error.

Usage::

    from agentprobe import BenchmarkConfig, BenchmarkRunner, ToolFailureConfig, ToolFailureType
    from agentprobe.agents.langgraph_agent import LangGraphToolAgent
    from agentprobe.fake_llm import ScriptedToolCallingModel

    llm = ScriptedToolCallingModel()
    agent = LangGraphToolAgent(
        llm=llm,
        failure_config=ToolFailureConfig(
            failure_type=ToolFailureType.RATE_LIMIT, failure_probability=0.3, seed=42
        ),
    )
    results = BenchmarkRunner(agent=agent, config=BenchmarkConfig(n_runs=10)).run()
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool

_ERROR_PREFIXES = ("error", "calculation error", "tool call", "rate limit", "http")


class ScriptedToolCallingModel(BaseChatModel):
    """Deterministic chat model that calls one tool, then answers from its result.

    Attributes:
        tool_name: Tool to call on the first turn. Falls back to the first bound tool
            if this name is not among the bound tools.
        tool_args: Arguments for that call.
        answer_template: Final answer; ``{result}`` is replaced by the tool output.
        error_template: Final answer when the tool output looks like an error.
        latency_ms: Optional simulated think time per call (sleep), for latency demos.
    """

    tool_name: str = "calculator"
    tool_args: dict[str, Any] = {"expression": "15 * 847 / 100"}
    answer_template: str = "The result is {result}."
    error_template: str = "I could not complete the task because the tool failed: {result}"
    latency_ms: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-calling"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.latency_ms > 0:
            import time

            time.sleep(self.latency_ms / 1000.0)

        last = messages[-1] if messages else None
        if isinstance(last, ToolMessage):
            result = str(last.content).strip()
            template = (
                self.error_template
                if result.lower().startswith(_ERROR_PREFIXES)
                else self.answer_template
            )
            message = AIMessage(content=template.format(result=result))
        else:
            bound = kwargs.get("tools") or []
            names = [
                t.get("function", {}).get("name")
                for t in bound
                if isinstance(t, dict) and t.get("function")
            ]
            name = self.tool_name if (self.tool_name in names or not names) else names[0]
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": name,
                        "args": dict(self.tool_args),
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


__all__ = ["ScriptedToolCallingModel"]
