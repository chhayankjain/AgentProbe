"""LangChain agent for AgentProbe benchmarking.

Uses LangChain v1.3+ ``create_agent`` (which wraps LangGraph internally)
with the same tools and prompts as the LangGraph agent, enabling
apples-to-apples framework comparison.

Usage::

    from langchain_ollama import ChatOllama
    from agentprobe.agents.langchain_agent import LangChainToolAgent
    from agentprobe.injector.tool_failure import ToolFailureConfig, ToolFailureType

    llm = ChatOllama(model="llama3.1")
    agent = LangChainToolAgent(
        llm=llm,
        failure_config=ToolFailureConfig(
            failure_type=ToolFailureType.RATE_LIMIT,
            failure_probability=0.15,
        ),
    )
    result = agent.invoke({"input": "What is 2^32?"})
"""

from __future__ import annotations

import time
from typing import Any

from agentprobe._extras import require

try:
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.tools import tool

    from agentprobe.agents.base import message_text
    from agentprobe.injector.tool_failure import ToolFailureConfig, ToolFailureInjector
    from agentprobe.llm import llm_call_boundary
    from agentprobe.observer.tracer import get_tracer
except ImportError:  # name the extra that covers every dependency, then re-raise
    require(
        "agentprobe.agents.langchain_agent",
        "research",
        "langchain",
        "langchain_core",
        "pydantic",
        "opentelemetry",
    )
    raise

_tracer = get_tracer(__name__)

_SYSTEM_PROMPT = (
    "You are a reliable AI assistant. Use the provided tools to answer questions "
    "accurately. If a tool call fails, acknowledge the failure and try a different approach."
)


class _LLMCallBoundaryMiddleware(AgentMiddleware):
    """Tags exceptions raised by the model call (not by tools) so the runner can
    tell provider failures apart from tool failures that share a transport."""

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        with llm_call_boundary():
            return handler(request)

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        with llm_call_boundary():
            return await handler(request)


def _make_tools(injector: ToolFailureInjector | None = None) -> list[Any]:
    @tool
    def web_search(query: str) -> str:
        """Search the web for current information."""
        return f"[MOCK WEB SEARCH] Results for: {query}\nFound 3 relevant results."

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            allowed = set("0123456789+-*/()., ")
            if not all(c in allowed for c in expression):
                return "Error: invalid characters in expression"
            result = eval(expression, {"__builtins__": {}})  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Calculation error: {e}"

    tools = [web_search, calculator]
    if injector is not None:
        tools = [injector.inject_langchain_tool(t) for t in tools]
    return tools


class LangChainToolAgent:
    """LangChain create_agent with AgentProbe instrumentation.

    Uses LangChain v1.3+ ``create_agent`` which returns a compiled
    LangGraph under the hood but exposes the LangChain tool/prompt API.

    Args:
        llm: A LangChain chat model supporting tool calling.
        failure_config: Optional tool failure injection configuration.
        recursion_limit: Maximum agent loop iterations.
    """

    framework = "langchain"

    def __init__(
        self,
        llm: Any,
        failure_config: ToolFailureConfig | None = None,
        recursion_limit: int = 25,
    ) -> None:
        self._llm = llm
        # Always present: with no fault configured it is a pass-through (type NONE never
        # injects), but it still counts tool calls so baseline trials carry evidence.
        self._injector = ToolFailureInjector(failure_config or ToolFailureConfig())
        self._tools = _make_tools(self._injector)
        self._agent = create_agent(
            self._llm,
            self._tools,
            system_prompt=_SYSTEM_PROMPT,
            middleware=[_LLMCallBoundaryMiddleware()],
        )
        self._recursion_limit = recursion_limit

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        with _tracer.start_as_current_span(
            "langchain.agent.invoke",
            attributes={"framework": self.framework},
        ) as span:
            start = time.perf_counter()
            query = str(input.get("input", ""))
            if context := input.get("context"):
                query = f"Context:\n{context}\n\nQuestion: {query}"

            result = self._agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                {"recursion_limit": self._recursion_limit},
            )
            latency_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("latency_ms", latency_ms)

            # Extract last AI message content
            messages = result.get("messages", [])
            output = ""
            for msg in reversed(messages):
                if type(msg).__name__ == "AIMessage":
                    text = message_text(msg)
                    if text:
                        output = text
                        break

            return {
                "output": output,
                "latency_ms": latency_ms,
                "framework": self.framework,
            }

    def reset(self) -> None:
        """No-op: each ``invoke()`` starts a new message list. Injector state is
        intentionally preserved (condition-wide); use ``agent.injector.reset()``.
        """

    @property
    def injector(self) -> ToolFailureInjector | None:
        return self._injector

    def get_injector_summary(self) -> dict[str, Any]:
        if self._injector:
            return self._injector.summary()
        return {}
