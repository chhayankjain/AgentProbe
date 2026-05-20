"""LangChain ReAct agent for AgentProbe benchmarking.

Uses LangChain's AgentExecutor with the same tools and prompts as the
LangGraph agent, enabling apples-to-apples framework comparison.

Usage::

    from langchain_ollama import ChatOllama
    from agentprobe.agents.langchain_agent import LangChainToolAgent
    from agentprobe.injector.tool_failure import ToolFailureConfig, ToolFailureType

    llm = ChatOllama(model="llama3")
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

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from agentprobe.injector.tool_failure import ToolFailureConfig, ToolFailureInjector
from agentprobe.observer.tracer import get_tracer

_tracer = get_tracer(__name__)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a reliable AI assistant. Use the provided tools to answer questions "
        "accurately. If a tool call fails, acknowledge the failure and try a different approach."
    )),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


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
    """LangChain AgentExecutor with AgentProbe instrumentation.

    Args:
        llm: A LangChain chat model supporting tool calling.
        failure_config: Optional tool failure injection configuration.
        max_iterations: Maximum ReAct iterations.
        handle_parsing_errors: Whether to let AgentExecutor handle parsing errors.
    """

    framework = "langchain"

    def __init__(
        self,
        llm: Any,
        failure_config: ToolFailureConfig | None = None,
        max_iterations: int = 10,
        handle_parsing_errors: bool = True,
    ) -> None:
        self._llm = llm
        self._injector = ToolFailureInjector(failure_config) if failure_config else None
        self._tools = _make_tools(self._injector)
        self._executor = self._build_executor(max_iterations, handle_parsing_errors)

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        with _tracer.start_as_current_span(
            "langchain.agent.invoke",
            attributes={"framework": self.framework},
        ) as span:
            start = time.perf_counter()
            query = str(input.get("input", ""))
            if context := input.get("context"):
                query = f"Context:\n{context}\n\nQuestion: {query}"

            result = self._executor.invoke({"input": query})
            latency_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("latency_ms", latency_ms)

            return {
                "output": result.get("output", ""),
                "latency_ms": latency_ms,
                "framework": self.framework,
            }

    def get_injector_summary(self) -> dict[str, Any]:
        if self._injector:
            return self._injector.summary()
        return {}

    def _build_executor(self, max_iterations: int, handle_parsing_errors: bool) -> AgentExecutor:
        agent = create_tool_calling_agent(self._llm, self._tools, _PROMPT)
        return AgentExecutor(
            agent=agent,
            tools=self._tools,
            max_iterations=max_iterations,
            handle_parsing_errors=handle_parsing_errors,
            verbose=False,
        )
