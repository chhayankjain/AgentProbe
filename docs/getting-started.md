---
description: Install AgentProbe, write your first fault test for a tool-using LLM agent, and read the failure report.
---

# Getting started

This page takes about five minutes and needs no model or API key.

## Install

```bash
pip install "agentprobe-toolkit>=1.4" pytest
```

You install `agentprobe-toolkit` and import `agentprobe`. Python 3.11 or later. From the next release, this installs nothing else: `agentprobe.testing` has no required dependencies. Optional extras add LangChain support (`[langchain]`), tracing export (`[observability]`) and the research toolkit (`[research]`).

## The one requirement

AgentProbe needs your agent to **receive its tools**, as an argument, a constructor parameter or a graph you build from a list. A test then hands the agent probed copies instead of the originals. If your agent imports its tools at module level, see the [case study](case-studies/agents-from-scratch.md) for the module-swap pattern.

## A first fault test

Here is a tiny agent with two tools. It stands in for yours.

```python
from agentprobe.testing import probe


def payment_status(order_id: str) -> str:
    return f"Order {order_id} is confirmed and paid."


def send_receipt(order_id: str) -> str:
    return f"Receipt sent for order {order_id}."


TOOLS = {"payment_status": payment_status, "send_receipt": send_receipt}


def answer(order_id: str, tools=TOOLS) -> str:
    try:
        status = tools["payment_status"](order_id)
    except Exception:
        return f"I couldn't confirm order {order_id} right now."
    tools["send_receipt"](order_id)
    return status
```

The test breaks `payment_status` on every call and checks that the agent admits it:

```python
def test_admits_failure_when_payment_service_is_down():
    with probe() as p:
        tools = p.wrap(TOOLS)
        p.on("payment_status").always_fail(with_="timeout")
        result = p.run(lambda q: answer(q, tools), "42")
        result.assert_no_crash()
        result.assert_answer_contains("couldn't confirm")
        result.assert_answer_not_contains("is confirmed")
        result.assert_not_called("send_receipt")
```

Run it with `pytest`. Three things happen:

1. `p.wrap(TOOLS)` returns probed copies. Your original functions are never modified.
2. `p.on("payment_status").always_fail(with_="timeout")` declares a **scenario**: every call to that tool raises an injected timeout instead of running.
3. When the `with` block ends, the **gate** checks that every required scenario fired and every required assertion passed. If the agent never called `payment_status`, the test fails even though every assertion passed.

## Reading a failure

Point the same test at an agent that swallows the error and reports success anyway, and pytest shows what happened, what you asserted and what AgentProbe inferred:

```text
AssertionError: answer does not contain 'is confirmed': answer was 'Order 42 is confirmed and paid; receipt sent for order 42.'

Observed
  1. payment_status('42') raised ToolCallTimeout: Tool call timed out (injected by AgentProbe)  [injected: payment_status: fail every call with timeout]
  2. send_receipt('42') returned 'receipt sent for order 42'
  scenarios:
    [exercised] payment_status: fail every call with timeout (required; injected 1/open-ended, 1 call(s) to payment_status)
Asserted
  [fail] answer does not contain 'is confirmed': answer was 'Order 42 is confirmed and paid; receipt sent for order 42.'
Inferred (heuristics over the call log)
  fallbacks: [('payment_status', 'send_receipt')]
```

An agent that never calls the tool fails the gate:

```text
AgentProbe gate FAILED: scenarios 0/1 required exercised, assertions 0/0 required passed
  - scenario 'payment_status: fail every call with timeout' not exercised: the agent never called 'payment_status' (tools called: none). It may have bypassed the tool or used a cached answer.
```

## Before you wire in real tools

- **Real tools really run.** Exception and substituted-return faults (`fail_*`, `always_fail`, `returns`) bypass your tool. Every other call runs it, including calls delayed by `slow()`. If a tool sends email, charges a card or writes to a database, point it at a sandbox or a test double.
- **Reports are not redacted yet.** Summaries and `to_dict()` include tool arguments and answers verbatim. Keep secrets out of fault-test inputs ([#102](https://github.com/chhayankjain/agentprobe-toolkit/issues/102)).

## Try the demo

The demo uses the research toolkit, so install it with the `research` extra first:

```bash
pip install "agentprobe-toolkit[research]"
python -m agentprobe.demo                       # rate-limit faults, p=0.3, 10 runs
python -m agentprobe.demo --fault timeout --p 0.5 --runs 20
```

The demo runs a small LangGraph agent on a scripted model and prints metrics and the per-call injection log.

## Next

- [Test that your agent retries](guides/retries.md)
- [Run fault tests in CI](guides/ci.md)
- The [10-minute tutorial](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/examples/your_agent/README.md) on GitHub, with a complete example agent and a deliberately broken version.
