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

from langchain.agents import create_agent
from langchain_core.tools import tool

from agentprobe.injector.tool_failure import ToolFailureConfig, ToolFailureInjector
from agentprobe.observer.tracer import get_tracer

_tracer = get_tracer(__name__)

_SYSTEM_PROMPT = (
    "You are a reliable AI assistant. Use the provided tools to answer questions "
    "accurately. If a tool call fails, acknowledge the failure and try a different approach."
)


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
        self._injector = ToolFailureInjector(failure_config) if failure_config else None
        self._tools = _make_tools(self._injector)
        self._agent = create_agent(
            self._llm,
            self._tools,
            system_prompt=_SYSTEM_PROMPT,
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
                if hasattr(msg, "content") and type(msg).__name__ == "AIMessage" and msg.content:
                    output = msg.content
                    break

            return {
                "output": output,
                "latency_ms": latency_ms,
                "framework": self.framework,
            }

    def get_injector_summary(self) -> dict[str, Any]:
        if self._injector:
            return self._injector.summary()
        return {}
