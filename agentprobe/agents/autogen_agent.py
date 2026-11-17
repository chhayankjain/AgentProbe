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

# Providers such as Gemini 3 attach provider extras (a ``thought_signature``
# under ``extra_content``) to each tool call and reject the next request unless
# the echoed function call carries them. pyautogen 0.1.14 stores only the legacy
# ``function_call`` {name, arguments}, so the shim remembers every full
# ``tool_calls`` entry *in response order* and restores the k-th occurrence of a
# (name, arguments) pair with the k-th remembered entry for that pair. The cache
# is cleared by ``AutoGenToolAgent.reset()`` at the start of each conversation,
# so identical calls in different trials never cross-contaminate. Entries without
# extras are never rewritten, so Ollama requests are byte-identical.
_TOOL_CALL_CACHE: dict[tuple[str, str], list[dict[str, Any]]] = {}
_TOOL_CALL_CACHE_MAX = 256

#: Optional client-side throttle applied around every ChatCompletion.create.
_RATE_LIMITER: Any = None


def _clear_tool_call_cache() -> None:
    _TOOL_CALL_CACHE.clear()


def _remember_tool_call(tool_call: dict[str, Any]) -> None:
    fn = tool_call.get("function", {})
    key = (fn.get("name", ""), fn.get("arguments", ""))
    total = sum(len(v) for v in _TOOL_CALL_CACHE.values())
    if total >= _TOOL_CALL_CACHE_MAX:
        _TOOL_CALL_CACHE.clear()
    _TOOL_CALL_CACHE.setdefault(key, []).append(dict(tool_call))


def _restore_tool_calls(messages: list[Any]) -> list[Any]:
    """Rewrite legacy assistant ``function_call`` messages whose original tool call
    carried provider extras back into ``tool_calls`` form. Others pass through.
    The k-th legacy occurrence of a (name, arguments) pair maps to the k-th
    remembered entry for that pair, so repeated identical calls keep their own
    ids and signatures."""
    out: list[Any] = []
    seen: dict[tuple[str, str], int] = {}
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("function_call"):
            fc = msg["function_call"]
            key = (fc.get("name", ""), fc.get("arguments", ""))
            idx = seen.get(key, 0)
            seen[key] = idx + 1
            entries = _TOOL_CALL_CACHE.get(key, [])
            cached = entries[idx] if idx < len(entries) else None
            if cached is not None and cached.get("extra_content"):
                rewritten = {k: v for k, v in msg.items() if k != "function_call"}
                rewritten["tool_calls"] = [cached]
                out.append(rewritten)
                continue
        out.append(msg)
    return out


def configure_compat(
    requests_per_minute: float | None = None, max_retries: int | None = None
) -> dict[str, Any]:
    """Apply the process-wide knobs pyautogen 0.1.14 exposes only globally.

    * ``requests_per_minute`` installs a token-bucket limiter that the shim
      acquires before every ``ChatCompletion.create``.
    * ``max_retries`` maps onto pyautogen's time-based retry window:
      ``Completion.max_retry_period = max_retries * Completion.retry_wait_time``
      (``0`` disables its internal retries so backend errors surface to the
      harness, which has its own retry policy).

    Returns the effective settings for metadata.
    """
    global _RATE_LIMITER
    effective: dict[str, Any] = {}
    if requests_per_minute is not None:
        from langchain_core.rate_limiters import InMemoryRateLimiter

        _RATE_LIMITER = InMemoryRateLimiter(
            requests_per_second=requests_per_minute / 60.0,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
        effective["requests_per_minute"] = requests_per_minute
    if max_retries is not None:
        try:
            from autogen import Completion

            Completion.max_retry_period = max_retries * Completion.retry_wait_time
            effective["max_retry_period_seconds"] = Completion.max_retry_period
        except ImportError:
            pass
        effective["max_retries"] = max_retries
    return effective


def _patch_openai_compat() -> None:
    """Monkey-patch ``openai.ChatCompletion.create`` to bridge the
    ``functions`` ↔ ``tools`` protocol gap between pyautogen 0.1.14
    and OpenAI-compatible endpoints (Ollama, Gemini, Groq, ...).

    Request:  ``functions`` list → ``tools`` list (wrapped in type/function);
              assistant ``function_call`` → ``tool_calls`` when the original
              call carried provider extras (thought signatures);
              optional requests-per-minute throttle (see :func:`configure_compat`).
    Response: ``tool_calls`` list → single ``function_call`` dict.

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
        if isinstance(kwargs.get("messages"), list):
            kwargs["messages"] = _restore_tool_calls(kwargs["messages"])
        if _RATE_LIMITER is not None:
            _RATE_LIMITER.acquire(blocking=True)

        resp = _original_create(*args, **kwargs)

        # --- response: tool_calls → function_call ---
        for choice in resp.get("choices", []):
            msg = choice.get("message", {})
            tool_calls = msg.get("tool_calls")
            if tool_calls and not msg.get("function_call"):
                first = tool_calls[0]
                _remember_tool_call(first)
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
        stop_on_final_answer: End the conversation at the first assistant
            message that is plain text (no function call). ``False`` (the
            paper's setting) lets pyautogen's user proxy keep auto-replying
            with empty messages until ``max_turns``. Required for providers
            that reject requests ending with a model turn (Gemini); see
            :func:`agentprobe.llm.autogen_agent_kwargs`.
        requests_per_minute: Client-side throttle applied to every LLM call
            (process-wide, since pyautogen 0.1.14 has no per-client hook).
        max_retries: Mapped onto pyautogen's time-based retry window; see
            :func:`configure_compat`.
    """

    framework = "autogen"

    def __init__(
        self,
        llm_config: dict[str, Any],
        failure_config: ToolFailureConfig | None = None,
        max_turns: int = 5,
        stop_on_final_answer: bool = False,
        requests_per_minute: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._llm_config = llm_config
        self._max_turns = max_turns
        self._stop_on_final_answer = stop_on_final_answer
        self._effective = configure_compat(requests_per_minute, max_retries)
        self._effective.update(
            {
                "temperature": llm_config.get("temperature"),
                "request_timeout_seconds": llm_config.get("request_timeout"),
                "stop_on_final_answer": stop_on_final_answer,
            }
        )
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
        _clear_tool_call_cache()  # per-conversation identity for restored tool calls
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

    def effective_settings(self) -> dict[str, Any]:
        """The LLM settings this adapter actually applies (for experiment metadata)."""
        return dict(self._effective)

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
            def _is_final_answer(message: dict[str, Any]) -> bool:
                return not message.get("function_call") and bool(
                    (message.get("content") or "").strip()
                )

            user_proxy = autogen.UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=self._max_turns,
                code_execution_config=False,
                is_termination_msg=_is_final_answer if self._stop_on_final_answer else None,
                function_map={
                    "web_search": _web_search,
                    "calculator": _calculator,
                },
            )

            return user_proxy, assistant

        except ImportError:
            # Deferred — ImportError raised at invoke() time instead
            return None, None  # type: ignore[return-value]
