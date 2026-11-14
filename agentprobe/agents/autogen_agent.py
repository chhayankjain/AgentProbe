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

**Version pin:** this adapter targets ``pyautogen==0.1.14`` with ``openai<1.0``
(install with ``pip install "agentprobe[autogen]"``). Modern AutoGen releases
(``ag2`` / ``autogen-agentchat`` 0.4+) have a different API and are not yet
supported; see the project roadmap.

**Protocol fix (Nov 2026):** pyautogen 0.1.14 uses the legacy OpenAI
``functions`` / ``function_call`` format, but Ollama 0.31+ only handles
the newer ``tools`` / ``tool_calls`` format.  :func:`_patch_openai_compat`
installs a one-time monkey-patch on ``openai.ChatCompletion.create`` that
converts between the two formats transparently.

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

_COMPAT_PATCHED = False


def _patch_openai_compat() -> None:
    """Monkey-patch ``openai.ChatCompletion.create`` to bridge the
    ``functions`` ↔ ``tools`` protocol gap between pyautogen 0.1.14
    and Ollama's OpenAI-compatible endpoint.

    Request:  ``functions`` list → ``tools`` list (wrapped in type/function)
    Response: ``tool_calls`` list → single ``function_call`` dict

    The patch is idempotent and applied at most once per process.
    """
    global _COMPAT_PATCHED
    if _COMPAT_PATCHED:
        return

    import openai

    _original_create = openai.ChatCompletion.create

    @staticmethod  # type: ignore[misc]
    def _patched_create(*args: Any, **kwargs: Any) -> Any:
        # --- request: functions → tools ---
        if "functions" in kwargs and "tools" not in kwargs:
            functions = kwargs.pop("functions")
            kwargs["tools"] = [{"type": "function", "function": f} for f in functions]
        # Also remove function_call if present (incompatible with tools)
        kwargs.pop("function_call", None)

        resp = _original_create(*args, **kwargs)

        # --- response: tool_calls → function_call ---
        for choice in resp.get("choices", []):
            msg = choice.get("message", {})
            tool_calls = msg.get("tool_calls")
            if tool_calls and not msg.get("function_call"):
                first = tool_calls[0]
                fn = first.get("function", {})
                msg["function_call"] = {
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "{}"),
                }
        return resp

    openai.ChatCompletion.create = _patched_create
    _COMPAT_PATCHED = True


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
        _patch_openai_compat()
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
            # Clear conversation history and response cache between runs
            self.reset()
            user_proxy.initiate_chat(
                assistant,
                message=query,
                max_turns=self._max_turns,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("latency_ms", latency_ms)

            # Extract last assistant message with nonempty content from
            # conversation history.  pyautogen 0.1.14 stores messages on the
            # agent objects, not the return value.  Tool-call messages have
            # role="assistant" but empty/None content, so we skip those to
            # find the actual natural-language response (if any).
            messages = assistant.chat_messages.get(user_proxy, [])
            output = ""
            for m in reversed(messages):
                if m.get("role") == "assistant":
                    content = (m.get("content") or "").strip()
                    if content:
                        output = content
                        break

            # Count tool executions from conversation history
            tool_calls = [m for m in messages if m.get("role") == "function"]
            span.set_attribute("tool_call_count", len(tool_calls))

            return {
                "output": output,
                "latency_ms": latency_ms,
                "framework": self.framework,
                "tool_calls": len(tool_calls),
            }

    def reset(self) -> None:
        """Clear both agents' conversation history and pyautogen's on-disk
        completion cache. Called automatically at the start of every
        ``invoke()``. Injector state is intentionally preserved (condition-wide).
        """
        user_proxy, assistant = self._agents
        if user_proxy is None or assistant is None:
            return  # pyautogen not installed; invoke() will raise a clear ImportError
        user_proxy.reset()
        assistant.reset()
        try:
            from autogen.oai.completion import Completion

            Completion.clear_cache()
        except Exception:
            pass

    @property
    def injector(self) -> ToolFailureInjector | None:
        return self._injector

    def get_injector_summary(self) -> dict[str, Any]:
        if self._injector:
            return self._injector.summary()
        return {}

    def _build_agents(self) -> tuple[Any, Any]:
        try:
            import autogen

            def _web_search(query: str) -> str:
                def _web_search_impl(q: str) -> str:
                    return f"[MOCK WEB SEARCH] Results for: {q}\nFound 3 relevant results."

                if self._injector:
                    return self._injector.call(
                        _web_search_impl,
                        query,
                        tool_name="web_search",
                    )
                return _web_search_impl(query)

            def _calculator(expression: str) -> str:
                def _calculator_impl(expr: str) -> str:
                    allowed = set("0123456789+-*/()., ")
                    if not all(c in allowed for c in expr):
                        return "Error: invalid characters"
                    try:
                        return str(eval(expr, {"__builtins__": {}}))  # noqa: S307
                    except Exception as e:
                        return f"Error: {e}"

                if self._injector:
                    return self._injector.call(
                        _calculator_impl,
                        expression,
                        tool_name="calculator",
                    )
                return _calculator_impl(expression)

            # Define function schemas for the LLM
            # These are in the legacy 'functions' format; _patch_openai_compat()
            # converts them to 'tools' format before sending to the API.
            functions = [
                {
                    "name": "web_search",
                    "description": "Search the web for current information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            }
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "calculator",
                    "description": "Evaluate a mathematical expression.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression to evaluate",
                            }
                        },
                        "required": ["expression"],
                    },
                },
            ]

            # Add function definitions to llm_config
            llm_config_with_functions = {
                **self._llm_config,
                "functions": functions,
            }

            assistant = autogen.AssistantAgent(
                name="assistant",
                llm_config=llm_config_with_functions,
                system_message=(
                    "You are a reliable AI assistant. Use the provided functions to "
                    "answer questions. If a tool call fails, acknowledge and try another approach."
                ),
            )

            # Register functions on the user_proxy so it can execute them
            user_proxy = autogen.UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=self._max_turns,
                code_execution_config=False,
                function_map={
                    "web_search": _web_search,
                    "calculator": _calculator,
                },
            )

            return user_proxy, assistant

        except ImportError:
            # Deferred — ImportError raised at invoke() time instead
            return None, None  # type: ignore[return-value]
