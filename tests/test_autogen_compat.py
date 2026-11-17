"""pyautogen compatibility shim: restoring provider extras on echoed tool calls.

Gemini 3 attaches a ``thought_signature`` (``extra_content``) to every tool call
and returns HTTP 400 if the next request echoes the call without it. pyautogen
0.1.14 only keeps the legacy ``function_call``; the shim restores the full entry.
"""

from __future__ import annotations

from agentprobe.agents import autogen_agent as ag


def _reset() -> None:
    ag._clear_tool_call_cache()


def test_extras_restored_on_echoed_function_call():
    _reset()
    original = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "calculator", "arguments": '{"expression": "2 + 2"}'},
        "extra_content": {"google": {"thought_signature": "sig=="}},
    }
    ag._remember_tool_call(original)
    legacy = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "calculator", "arguments": '{"expression": "2 + 2"}'},
        },
        {"role": "function", "name": "calculator", "content": "4"},
    ]
    out = ag._restore_tool_calls(legacy)
    assert out[0] == legacy[0] and out[2] == legacy[2]
    assert "function_call" not in out[1]
    assert out[1]["tool_calls"] == [original]
    assert out[1]["role"] == "assistant"


def test_ollama_path_untouched_when_no_extras():
    """Ollama returns no extras, so the legacy format the paper used is sent unchanged."""
    _reset()
    ag._remember_tool_call(
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expression": "1"}'},
        }
    )
    legacy = [
        {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "calculator", "arguments": '{"expression": "1"}'},
        }
    ]
    assert ag._restore_tool_calls(legacy) == legacy


def test_unknown_call_passes_through():
    _reset()
    legacy = [
        {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "web_search", "arguments": '{"query": "x"}'},
        }
    ]
    assert ag._restore_tool_calls(legacy) == legacy


def test_cache_is_bounded():
    _reset()
    for i in range(ag._TOOL_CALL_CACHE_MAX + 10):
        ag._remember_tool_call(
            {"function": {"name": "f", "arguments": str(i)}, "extra_content": {}}
        )
    assert sum(len(v) for v in ag._state().cache.values()) <= ag._TOOL_CALL_CACHE_MAX


def test_stop_on_final_answer_sets_termination_predicate():
    import pytest

    pytest.importorskip("autogen")
    from agentprobe.agents.autogen_agent import AutoGenToolAgent

    cfg = {"config_list": [{"model": "m", "api_base": "http://x/v1", "api_key": "k"}]}
    default = AutoGenToolAgent(llm_config=cfg)
    gemini = AutoGenToolAgent(llm_config=cfg, stop_on_final_answer=True)
    # pyautogen's default predicate only stops on a literal "TERMINATE".
    default_pred = default._agents[0]._is_termination_msg
    assert default_pred({"content": "The answer is 4."}) is False
    pred = gemini._agents[0]._is_termination_msg
    assert pred({"content": "The answer is 4."}) is True
    assert pred({"content": "", "function_call": {"name": "calculator"}}) is False
    assert pred({"content": None}) is False
