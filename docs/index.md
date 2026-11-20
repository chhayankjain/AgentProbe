---
title: Test how your LLM agent handles tool failures
hide:
  - navigation
description: AgentProbe is a Python library for fault-injection testing of tool-using LLM agents in pytest. Break tools deterministically and check what the agent did.
---

# Test how your LLM agent handles tool failures

AgentProbe is a pytest library for fault-injection testing of tool-using LLM agents.
{ .lead }

Your agent's tools will time out, hit rate limits and return bad data. AgentProbe wraps the tools your agent already uses, breaks them on purpose and deterministically, and checks what your agent did about it.

A test fails unless every fault you declared **actually fired**. An agent that skips the tool or answers from a cache can't pass a fault test by accident.

```bash
pip install "agentprobe-toolkit>=1.5" pytest
```

```py
from agentprobe.testing import probe

from my_agent import TOOLS, answer


def test_admits_failure_when_payment_service_is_down():
    with probe() as p:
        tools = p.wrap(TOOLS)  # probed copies; originals untouched
        p.on("payment_status").always_fail(with_="timeout")
        result = p.run(lambda q: answer(q, tools), "Did order 42 go through?")
        result.assert_no_crash()
        result.assert_answer_not_contains("is confirmed")  # no false success
        result.assert_not_called("send_receipt")  # no receipt if unconfirmed
```

Nothing is mocked except the fault: your agent runs for real, with real tools, and AgentProbe only steps in at the tool boundary. No model or API key is needed for the examples.

## Where to go next

<div class="grid cards" markdown>

- **[Getting started](getting-started.md)**: install, write a first fault test and read a failure report.
- **[Guides](guides/retries.md)**: retries, timeouts and rate limits, wrong data, LangGraph tool errors, CI.
- **[Reference](reference/scenarios.md)**: every scenario, assertion and result field.
- **[Case study](case-studies/agents-from-scratch.md)**: fault tests on a third-party LangGraph agent found an unhandled tool exception.

</div>

## What it works with

- Plain Python functions, sync or `async`.
- LangChain tools (`@tool`, `StructuredTool`, `BaseTool` subclasses), wrapped as copies with the same schema.
- LangGraph agents: pass the wrapped tools to your graph.
- Any agent callable or object with `invoke()`, sync or async.

AgentProbe is MIT-licensed. Source, issues and discussions are on [GitHub](https://github.com/chhayankjain/agentprobe-toolkit).
