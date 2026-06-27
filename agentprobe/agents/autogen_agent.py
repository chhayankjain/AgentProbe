"""AutoGen agent for AgentProbe benchmarking.

Implements a two-agent AutoGen pipeline (UserProxy + AssistantAgent)
using the same tool set as LangGraph and LangChain agents for
apples-to-apples comparison.

The AutoGen agent uses the OpenAI-compatible API endpoint, which can
point to an Ollama local server:

    AUTOGEN_LLM_CONFIG = {
        "config_list": [
            {
                "model": "llama3",
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
            }
        ]
    }

Usage::

    from agentprobe.agents.autogen_agent import AutoGenToolAgent

    agent = AutoGenToolAgent(
        llm_config=AUTOGEN_LLM_CONFIG,
        failure_config=ToolFailureConfig(
            failure_type=ToolFailureType.MALFORMED_OUTPUT,
            failure_probability=0.1,
        ),
    )
    result = agent.invoke({"input": "Find the top 3 LLM papers from 2024."})
"""

from __future__ import annotations

import time
from typing import Any

from agentprobe.injector.tool_failure import ToolFailureConfig, ToolFailureInjector
from agentprobe.observer.tracer import get_tracer

_tracer = get_tracer(__name__)


class AutoGenToolAgent:
    """AutoGen AssistantAgent + UserProxy with AgentProbe instrumentation.

    Falls back gracefully if pyautogen is not installed, logging a clear
    ImportError with install instructions.

    Args:
        llm_config: AutoGen llm_config dict (see AutoGen docs).
        failure_config: Optional tool failure injection configuration.
        max_turns: Maximum conversation turns between agents.
    """

    framework = "autogen"

    def __init__(
        self,
        llm_config: dict[str, Any],
        failure_config: ToolFailureConfig | None = None,
        max_turns: int = 5,
    ) -> None:
        self._llm_config = llm_config
        self._max_turns = max_turns
        self._injector = ToolFailureInjector(failure_config) if failure_config else None
        self._agents = self._build_agents()

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        try:
            import autogen  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pyautogen is not installed. Install with: pip install pyautogen"
            ) from e

        with _tracer.start_as_current_span(
            "autogen.agent.invoke",
            attributes={"framework": self.framework},
        ) as span:
            start = time.perf_counter()
            query = str(input.get("input", ""))
            if context := input.get("context"):
                query = f"Context:\n{context}\n\nQuestion: {query}"

            user_proxy, assistant = self._agents
            chat_result = user_proxy.initiate_chat(
                assistant,
                message=query,
                max_turns=self._max_turns,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("latency_ms", latency_ms)

            # Extract last assistant message as output
            messages = getattr(chat_result, "chat_history", []) or []
            output = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"),
                "",
            )
            return {
                "output": output,
                "latency_ms": latency_ms,
                "framework": self.framework,
            }

    def get_injector_summary(self) -> dict[str, Any]:
        if self._injector:
            return self._injector.summary()
        return {}

    def _build_agents(self) -> tuple[Any, Any]:
        try:
            import autogen

            def _web_search(query: str) -> str:
                fn = lambda q: f"[MOCK WEB SEARCH] Results for: {q}\nFound 3 relevant results."  # noqa: E731
                if self._injector:
                    return self._injector.call(fn, query)
                return fn(query)

            def _calculator(expression: str) -> str:
                def fn(expr: str) -> str:
                    allowed = set("0123456789+-*/()., ")
                    if not all(c in allowed for c in expr):
                        return "Error: invalid characters"
                    try:
                        return str(eval(expr, {"__builtins__": {}}))  # noqa: S307
                    except Exception as e:
                        return f"Error: {e}"

                if self._injector:
                    return self._injector.call(fn, expression)
                return fn(expression)

            assistant = autogen.AssistantAgent(
                name="assistant",
                llm_config=self._llm_config,
                system_message=(
                    "You are a reliable AI assistant. Use the provided functions to "
                    "answer questions. If a tool call fails, acknowledge and try another approach."
                ),
            )

            user_proxy = autogen.UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=self._max_turns,
                code_execution_config=False,
            )

            # Register tools
            autogen.register_function(
                _web_search,
                caller=assistant,
                executor=user_proxy,
                name="web_search",
                description="Search the web for current information.",
            )
            autogen.register_function(
                _calculator,
                caller=assistant,
                executor=user_proxy,
                name="calculator",
                description="Evaluate a mathematical expression.",
            )

            return user_proxy, assistant

        except ImportError:
            # Deferred — ImportError raised at invoke() time instead
            return None, None  # type: ignore[return-value]
