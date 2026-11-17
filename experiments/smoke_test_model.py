"""Quick smoke test: verify a model can call tools via each framework.

Usage:
    .venv/bin/python experiments/smoke_test_model.py --model mistral:7b
    .venv/bin/python experiments/smoke_test_model.py --model qwen2.5:7b
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentprobe.injector.tool_failure import ToolFailureConfig, ToolFailureType
from agentprobe.llm import add_llm_args, autogen_agent_kwargs, autogen_llm_config, build_chat_model, spec_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test model tool calling")
    add_llm_args(parser, default_model="llama3.1")
    parser.add_argument("--frameworks", nargs="+",
                       default=["langgraph", "langchain", "autogen"],
                       choices=["langgraph", "langchain", "autogen"])
    args = parser.parse_args()
    spec = spec_from_args(args)
    if not spec.is_local:
        spec.resolve_api_key()
    print(f"Backend: {spec.provider} · {spec.model} · {spec.resolved_base_url}")

    config = ToolFailureConfig(
        failure_type=ToolFailureType.NONE,
        failure_probability=0.0,
        seed=42,
    )

    query = (
        "You MUST call the calculator tool with the expression '2 + 2' "
        "and report the exact numeric result. Do NOT compute this yourself."
    )

    for fw in args.frameworks:
        print(f"\n--- {fw} / {args.model} ---")
        start = time.perf_counter()
        try:
            if fw in ("langgraph", "langchain"):
                llm = build_chat_model(spec)
                if fw == "langgraph":
                    from agentprobe.agents.langgraph_agent import LangGraphToolAgent
                    agent = LangGraphToolAgent(llm=llm, failure_config=config)
                else:
                    from agentprobe.agents.langchain_agent import LangChainToolAgent
                    agent = LangChainToolAgent(llm=llm, failure_config=config)
            else:
                from agentprobe.agents.autogen_agent import AutoGenToolAgent
                agent = AutoGenToolAgent(
                    llm_config=autogen_llm_config(spec),
                    failure_config=config,
                    **autogen_agent_kwargs(spec),
                )

            result = agent.invoke({"input": query})
            elapsed = time.perf_counter() - start
            output = str(result.get("output", ""))

            # Check tool execution
            log = agent.injector.injection_log if agent.injector else []
            tool_names = [r.tool_name for r in log]
            calc_called = any(r.tool_name == "calculator" for r in log)
            calc_return = next(
                (str(r.return_value) for r in log if r.tool_name == "calculator" and r.return_value),
                None
            )

            status = "PASS" if calc_called and "4" in output else "FAIL"
            color = "\033[92m" if status == "PASS" else "\033[91m"
            print(f"  {color}{status}\033[0m ({elapsed:.1f}s)")
            print(f"  Output: {output[:120]}")
            print(f"  Tools called: {tool_names}")
            print(f"  Calculator return: {calc_return}")
            print(f"  Answer '4' in output: {'4' in output}")

        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"  \033[91mERROR\033[0m ({elapsed:.1f}s): {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
