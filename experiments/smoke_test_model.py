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


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test model tool calling")
    parser.add_argument("--model", required=True)
    parser.add_argument("--frameworks", nargs="+",
                       default=["langgraph", "langchain", "autogen"],
                       choices=["langgraph", "langchain", "autogen"])
    args = parser.parse_args()

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
                from langchain_ollama import ChatOllama
                llm = ChatOllama(model=args.model, temperature=0.0)
                if fw == "langgraph":
                    from agentprobe.agents.langgraph_agent import LangGraphToolAgent
                    agent = LangGraphToolAgent(llm=llm, failure_config=config)
                else:
                    from agentprobe.agents.langchain_agent import LangChainToolAgent
                    agent = LangChainToolAgent(llm=llm, failure_config=config)
            else:
                from agentprobe.agents.autogen_agent import AutoGenToolAgent
                llm_config = {
                    "config_list": [{
                        "model": args.model,
                        "api_base": "http://localhost:11434/v1",
                        "api_key": "ollama",
                    }]
                }
                agent = AutoGenToolAgent(llm_config=llm_config, failure_config=config)

            result = agent.invoke({"input": query})
            elapsed = time.perf_counter() - start
            output = str(result.get("output", ""))

            # Check tool execution
            log = agent._injector.injection_log if agent._injector else []
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
