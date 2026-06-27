"""LangGraph ReAct agent for AgentProbe benchmarking.

Implements a standard LangGraph ReAct agent with:
  - Configurable LLM backend (Ollama/OpenAI/Anthropic)
  - Two benchmark tools: web_search (mocked) and calculator
  - Full OpenTelemetry instrumentation on every node and tool call
  - Failure injector hooks on tool execution

The graph structure:
    [START] -> agent_node -> tool_node -> agent_node -> ... -> [END]

Usage::

    from langchain_ollama import ChatOllama
    from agentprobe.agents.langgraph_agent import LangGraphToolAgent
    from agentprobe.injector.tool_failure import ToolFailureConfig, ToolFailureType

    llm = ChatOllama(model="llama3")
    agent = LangGraphToolAgent(
        llm=llm,
        failure_config=ToolFailureConfig(
            failure_type=ToolFailureType.TIMEOUT,
            failure_probability=0.2,
        ),
    )
    result = agent.invoke({"input": "What is the capital of France?"})
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agentprobe.injector.tool_failure import ToolFailureConfig, ToolFailureInjector
from agentprobe.observer.tracer import get_tracer

_tracer = get_tracer(__name__)

SYSTEM_PROMPT = (
    "You are a reliable AI assistant. Use the provided tools to answer questions "
    "accurately. If a tool call fails, acknowledge the failure and try a different approach."
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Tools (mockable for offline benchmarking)
# ---------------------------------------------------------------------------


def _make_tools(injector: ToolFailureInjector | None = None) -> list[Any]:
    """Create benchmark tools, optionally wrapping with failure injector."""

    @tool
    def web_search(query: str) -> str:
        """Search the web for current information."""
        # Mock response — replace with real search in production experiments
        return f"[MOCK WEB SEARCH] Results for: {query}\nFound 3 relevant results."

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            # Safe eval: only allow numeric expressions
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


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class LangGraphToolAgent:
    """LangGraph ReAct agent with AgentProbe instrumentation.

    Args:
        llm: A LangChain chat model (ChatOllama, ChatOpenAI, ChatAnthropic, etc.)
        failure_config: Optional tool failure injection configuration.
        max_iterations: Maximum ReAct loop iterations before forced termination.
    """

    framework = "langgraph"

    def __init__(
        self,
        llm: Any,
        failure_config: ToolFailureConfig | None = None,
        max_iterations: int = 10,
    ) -> None:
        self._llm = llm
        self._max_iterations = max_iterations
        self._injector = ToolFailureInjector(failure_config) if failure_config else None
        self._tools = _make_tools(self._injector)
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        """Run the agent on a task input. Returns output dict."""
        with _tracer.start_as_current_span(
            "langgraph.agent.invoke",
            attributes={
                "framework": self.framework,
                "input.length": len(str(input.get("input", ""))),
            },
        ) as span:
            start = time.perf_counter()
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=str(input.get("input", ""))),
            ]
            # Add RAG context if provided
            if context := input.get("context"):
                messages.insert(1, SystemMessage(content=f"Context:\n{context}"))

            state: AgentState = {"messages": messages}
            result = self._graph.invoke(
                state,
                config={"recursion_limit": self._max_iterations},
            )
            latency_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("latency_ms", latency_ms)

            final_message = result["messages"][-1]
            return {
                "output": getattr(final_message, "content", str(final_message)),
                "latency_ms": latency_ms,
                "framework": self.framework,
            }

    def get_injector_summary(self) -> dict[str, Any]:
        if self._injector:
            return self._injector.summary()
        return {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        llm_with_tools = self._llm.bind_tools(self._tools)
        tool_node = ToolNode(self._tools)

        def agent_node(state: AgentState) -> AgentState:
            with _tracer.start_as_current_span("langgraph.node.agent"):
                response = llm_with_tools.invoke(state["messages"])
                return {"messages": [response]}

        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        builder = StateGraph(AgentState)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", tool_node)
        builder.set_entry_point("agent")
        builder.add_conditional_edges("agent", should_continue)
        builder.add_edge("tools", "agent")

        return builder.compile()
