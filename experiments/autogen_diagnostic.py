"""Traced AutoGen diagnostic: minimal client vs AutoGen shim.

Runs the same deterministic tasks through:
  1. A minimal OpenAI-compatible client (direct /v1/chat/completions)
  2. The AutoGen shim (pyautogen 0.1.14 + _patch_openai_compat)

Captures and saves:
  - Full raw request/response payloads
  - All tool calls (name, arguments, result)
  - Final-answer selection and termination reason
  - Message-level conversation trace

Outputs a JSON diagnostic artifact for comparison.

Usage::

    .venv/bin/python experiments/autogen_diagnostic.py --model llama3.1
    .venv/bin/python experiments/autogen_diagnostic.py --model qwen2.5:7b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Shared tool definitions (identical for both paths)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
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
    },
    {
        "type": "function",
        "function": {
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
    },
]

# Same tools in legacy "functions" format (for AutoGen)
LEGACY_FUNCTIONS = [s["function"] for s in TOOL_SCHEMAS]

SYSTEM_PROMPT = (
    "You are a reliable AI assistant. Use the provided functions to "
    "answer questions. If a tool call fails, acknowledge and try another approach."
)

# Deterministic tasks with known correct answers
TASKS = [
    {
        "prompt": "What is the capital of France?",
        "expected_tool": "web_search",
        "expected_answer": "paris",
    },
    {
        "prompt": "Calculate 15% of 847.",
        "expected_tool": "calculator",
        "expected_answer": "127.05",
    },
    {
        "prompt": "Calculate 7919 * 6841.",
        "expected_tool": "calculator",
        "expected_answer": "54173879",
    },
]


def _check_answer_present(final_answer: str, expected: str) -> tuple[bool, bool]:
    """Check answer presence with both literal and normalized rules.

    Returns (literal_present, normalized_present).
    Normalized strips commas from both strings before comparing.
    """
    final = (final_answer or "").lower()
    exp = expected.lower()
    literal = exp in final
    normalized = exp.replace(",", "") in final.replace(",", "")
    return literal, normalized


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool call and return the result string."""
    if name == "calculator":
        expr = arguments.get("expression", "")
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expr):
            return "Error: invalid characters"
        try:
            return str(eval(expr, {"__builtins__": {}}))  # noqa: S307
        except Exception as e:
            return f"Error: {e}"
    elif name == "web_search":
        query = arguments.get("query", "")
        return f"[MOCK WEB SEARCH] Results for: {query}\nFound 3 relevant results."
    else:
        return f"Error: unknown tool '{name}'"


# ---------------------------------------------------------------------------
# Path 1: Minimal direct client
# ---------------------------------------------------------------------------

def run_direct_client(
    model: str,
    base_url: str,
    task: dict,
    max_turns: int = 5,
) -> dict[str, Any]:
    """Run a task through the OpenAI-compatible API directly.

    Implements a simple tool-use loop:
      1. Send messages with tools
      2. If response has tool_calls, execute them and send results back
      3. Repeat until the model responds with text or max_turns reached
    """
    import openai

    # Use legacy openai 0.28 API (same package version as pyautogen 0.1.14)
    old_base = openai.api_base
    old_key = openai.api_key
    openai.api_base = base_url
    openai.api_key = "ollama"

    trace: dict[str, Any] = {
        "path": "direct_client",
        "model": model,
        "task": task["prompt"],
        "turns": [],
        "tool_executions": [],
        "final_answer": None,
        "termination_reason": None,
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]},
    ]

    try:
      for turn_idx in range(max_turns):
        # Build request — use "tools" format (Ollama's native format)
        request_payload = {
            "model": model,
            "messages": deepcopy(messages),
            "tools": TOOL_SCHEMAS,
            "temperature": 0.0,
        }

        turn_record = {
            "turn": turn_idx,
            "request": deepcopy(request_payload),
        }

        # Call API using legacy openai.ChatCompletion.create
        start = time.perf_counter()
        try:
            response = openai.ChatCompletion.create(**request_payload)
            latency_ms = (time.perf_counter() - start) * 1000

            # response is already a dict-like object in openai 0.28
            resp_dict = dict(response)
            turn_record["response"] = resp_dict
            turn_record["latency_ms"] = latency_ms

        except Exception as e:
            turn_record["error"] = f"{type(e).__name__}: {e}"
            turn_record["latency_ms"] = (time.perf_counter() - start) * 1000
            trace["turns"].append(turn_record)
            trace["termination_reason"] = f"api_error: {e}"
            break

        choice = resp_dict["choices"][0]
        msg = choice["message"]
        finish_reason = choice.get("finish_reason", "unknown")

        # Check for tool calls
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            # Add assistant message with tool calls to conversation
            messages.append(dict(msg))

            for tc in tool_calls:
                fn = tc["function"]
                fn_name = fn["name"]
                try:
                    fn_args = json.loads(fn["arguments"])
                except (json.JSONDecodeError, TypeError):
                    fn_args = {"raw": fn["arguments"]}

                result = execute_tool(fn_name, fn_args)

                tool_record = {
                    "turn": turn_idx,
                    "tool_call_id": tc.get("id", f"call_{turn_idx}"),
                    "name": fn_name,
                    "arguments": fn_args,
                    "result": result,
                }
                trace["tool_executions"].append(tool_record)

                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{turn_idx}"),
                    "content": result,
                })

            trace["turns"].append(turn_record)
            continue  # Next turn — model should process tool results

        else:
            # Model responded with text — done
            content = msg.get("content", "")
            trace["final_answer"] = content
            trace["termination_reason"] = f"text_response (finish_reason={finish_reason})"
            trace["turns"].append(turn_record)
            break
      else:
        trace["termination_reason"] = f"max_turns_reached ({max_turns})"
        # Try to extract last content
        if trace["turns"]:
            last_msg = trace["turns"][-1].get("response", {})
            choices = last_msg.get("choices", [{}])
            if choices:
                trace["final_answer"] = choices[0].get("message", {}).get("content", "")
    finally:
        # Restore original openai settings
        openai.api_base = old_base
        openai.api_key = old_key

    trace["total_turns"] = len(trace["turns"])
    trace["total_tool_calls"] = len(trace["tool_executions"])

    # Check answer presence (both literal and comma-normalized)
    literal, normalized = _check_answer_present(
        trace["final_answer"], task["expected_answer"]
    )
    trace["answer_present"] = literal
    trace["answer_present_normalized"] = normalized

    return trace


# ---------------------------------------------------------------------------
# Path 2: AutoGen shim
# ---------------------------------------------------------------------------

def run_autogen_shim(
    model: str,
    base_url: str,
    task: dict,
    max_turns: int = 5,
) -> dict[str, Any]:
    """Run a task through the full AutoGen adapter pipeline.

    Captures the complete conversation history and all messages exchanged
    between UserProxy and AssistantAgent.
    """
    import autogen
    from agentprobe.agents.autogen_agent import _patch_openai_compat

    _patch_openai_compat()

    trace: dict[str, Any] = {
        "path": "autogen_shim",
        "model": model,
        "task": task["prompt"],
        "turns": [],
        "tool_executions": [],
        "final_answer": None,
        "termination_reason": None,
    }

    # Build tool functions (same implementations)
    def _web_search(query: str) -> str:
        result = execute_tool("web_search", {"query": query})
        trace["tool_executions"].append({
            "name": "web_search",
            "arguments": {"query": query},
            "result": result,
        })
        return result

    def _calculator(expression: str) -> str:
        result = execute_tool("calculator", {"expression": expression})
        trace["tool_executions"].append({
            "name": "calculator",
            "arguments": {"expression": expression},
            "result": result,
        })
        return result

    llm_config = {
        "config_list": [{
            "model": model,
            "api_base": base_url,
            "api_key": "ollama",
        }],
        "functions": LEGACY_FUNCTIONS,
        "temperature": 0.0,
    }

    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config=llm_config,
        system_message=SYSTEM_PROMPT,
    )

    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=max_turns,
        code_execution_config=False,
        function_map={
            "web_search": _web_search,
            "calculator": _calculator,
        },
    )

    # Clear caches
    try:
        from autogen.oai.completion import Completion
        Completion.clear_cache()
    except Exception:
        pass

    start = time.perf_counter()
    user_proxy.initiate_chat(
        assistant,
        message=task["prompt"],
        max_turns=max_turns,
    )
    total_latency_ms = (time.perf_counter() - start) * 1000

    # Extract full conversation history
    messages = assistant.chat_messages.get(user_proxy, [])

    for i, m in enumerate(messages):
        role = m.get("role", "unknown")
        content = m.get("content", "")
        function_call = m.get("function_call")
        name = m.get("name")

        turn_record: dict[str, Any] = {
            "index": i,
            "role": role,
            "content": content,
        }
        if function_call:
            turn_record["function_call"] = function_call
        if name:
            turn_record["name"] = name

        trace["turns"].append(turn_record)

    # Extract final answer (same logic as autogen_agent.py)
    output = ""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            c = (m.get("content") or "").strip()
            if c:
                output = c
                break

    trace["final_answer"] = output
    trace["total_latency_ms"] = total_latency_ms
    trace["total_turns"] = len(messages)
    trace["total_tool_calls"] = len(trace["tool_executions"])

    # Determine termination reason
    if not messages:
        trace["termination_reason"] = "no_messages"
    elif messages[-1].get("role") == "assistant" and (messages[-1].get("content") or "").strip():
        trace["termination_reason"] = "assistant_text_response"
    elif any(m.get("function_call") for m in messages[-2:] if m.get("role") == "assistant"):
        trace["termination_reason"] = "max_turns_with_pending_tool_call"
    else:
        trace["termination_reason"] = f"conversation_ended (last_role={messages[-1].get('role', '?')})"

    # Check answer presence (both literal and comma-normalized)
    literal, normalized = _check_answer_present(
        trace["final_answer"], task["expected_answer"]
    )
    trace["answer_present"] = literal
    trace["answer_present_normalized"] = normalized

    return trace


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AutoGen diagnostic: direct client vs shim")
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parent / "results_v2" / f"diagnostic_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"AutoGen diagnostic — model: {args.model}")
    print(f"Output: {out_dir}\n")

    # -----------------------------------------------------------------------
    # IMPORTANT: Run ALL direct-client tasks FIRST, before any AutoGen task.
    # The AutoGen shim calls _patch_openai_compat() which globally monkey-
    # patches openai.ChatCompletion.create. Running direct tasks after the
    # patch contaminates the control path with shim-added response fields.
    # -----------------------------------------------------------------------

    direct_results: dict[str, dict[str, Any]] = {}
    autogen_results: dict[str, dict[str, Any]] = {}

    print("=" * 50)
    print("Phase 1: Direct client (unpatched)")
    print("=" * 50)
    for task in TASKS:
        label = task["prompt"]
        print(f"  {label}...", end=" ", flush=True)
        try:
            direct = run_direct_client(args.model, args.base_url, task, args.max_turns)
            print(f"answer_present={direct['answer_present']}, "
                  f"tool_calls={direct['total_tool_calls']}")
        except Exception as e:
            direct = {"path": "direct_client", "error": f"{type(e).__name__}: {e}",
                      "task": label}
            print(f"ERROR: {e}")
        direct_results[label] = direct

    # Verify patch has NOT been applied yet
    from agentprobe.agents.autogen_agent import _COMPAT_PATCHED
    patch_applied_before_ag = _COMPAT_PATCHED

    print(f"\nPatch state before AutoGen phase: {patch_applied_before_ag}")
    print()

    print("=" * 50)
    print("Phase 2: AutoGen shim (patched, T=0)")
    print("=" * 50)
    for task in TASKS:
        label = task["prompt"]
        print(f"  {label}...", end=" ", flush=True)
        try:
            autogen_result = run_autogen_shim(args.model, args.base_url, task, args.max_turns)
            print(f"answer_present={autogen_result['answer_present']}, "
                  f"tool_calls={autogen_result['total_tool_calls']}")
        except Exception as e:
            autogen_result = {"path": "autogen_shim", "error": f"{type(e).__name__}: {e}",
                              "task": label}
            print(f"ERROR: {e}")
        autogen_results[label] = autogen_result

    all_results: list[dict[str, Any]] = []
    for task in TASKS:
        label = task["prompt"]
        all_results.append({
            "task": task,
            "direct_client": direct_results[label],
            "autogen_shim": autogen_results[label],
        })
    print()

    # Save full diagnostic
    diagnostic = {
        "metadata": {
            "model": args.model,
            "base_url": args.base_url,
            "max_turns": args.max_turns,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "execution_order": "all_direct_first_then_all_autogen",
            "direct_temperature": 0.0,
            "autogen_temperature": 0.0,
            "patch_applied_before_autogen_phase": patch_applied_before_ag,
        },
        "tasks": all_results,
    }

    out_path = out_dir / f"diagnostic_{args.model.replace(':', '_').replace('/', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(diagnostic, f, indent=2, default=str)
    print(f"Saved diagnostic to {out_path}")

    # Print comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Task':<30} {'Direct':>12} {'AutoGen':>12} {'Match':>8}")
    print("-" * 70)
    for r in all_results:
        task_label = r["task"]["prompt"][:28]
        d = r["direct_client"]
        a = r["autogen_shim"]
        d_ans = "YES" if d.get("answer_present") else "NO"
        a_ans = "YES" if a.get("answer_present") else "NO"
        d_tc = d.get("total_tool_calls", "?")
        a_tc = a.get("total_tool_calls", "?")
        match = "OK" if d.get("answer_present") == a.get("answer_present") else "DIFF"
        print(f"{task_label:<30} {d_ans}({d_tc}tc){' ':>3} {a_ans}({a_tc}tc){' ':>3} {match:>5}")

    # Check for protocol divergence
    print()
    for r in all_results:
        d = r["direct_client"]
        a = r["autogen_shim"]
        if d.get("answer_present") and not a.get("answer_present"):
            print(f"DIVERGENCE: '{r['task']['prompt'][:40]}' — direct client found answer, AutoGen did not")
            print(f"  Direct final: {(d.get('final_answer') or '')[:100]}")
            print(f"  AutoGen final: {(a.get('final_answer') or '')[:100]}")
            print(f"  AutoGen termination: {a.get('termination_reason')}")
            # Check if tool result reached the model
            if a.get("tool_executions"):
                last_tool = a["tool_executions"][-1]
                print(f"  Last tool result: {last_tool['name']}({last_tool['arguments']}) -> {last_tool['result'][:60]}")
            print()


if __name__ == "__main__":
    main()
