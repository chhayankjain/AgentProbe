---
description: Test that an LLM agent retries a failing tool call and recovers, with deterministic fault injection in pytest.
---

# Test that your agent retries

A transient failure should cost one retry, not the whole run. These tests make the first call to a tool fail, deterministically, and check that the agent recovers, and that it gives up honestly when retries run out.

## The agent under test

```python
from agentprobe.testing import probe


def payment_status(order_id: str) -> str:
    return f"Order {order_id} is paid."


TOOLS = {"payment_status": payment_status}


def answer(order_id: str, tools=TOOLS) -> str:
    for attempt in range(2):  # one retry
        try:
            return tools["payment_status"](order_id)
        except Exception:
            if attempt == 1:
                return f"I couldn't confirm order {order_id}."
```

## Recovers after one failure

`fail_first(1)` makes call 1 time out and lets later calls through to the real tool.

```python
def test_recovers_after_one_timeout():
    with probe() as p:
        tools = p.wrap(TOOLS)
        p.on("payment_status").fail_first(1)
        result = p.run(lambda q: answer(q, tools), "42")
        result.assert_no_crash()
        result.assert_answer_contains("is paid")

    first, retry = result.calls
    assert first.injected and first.exception_type == "ToolCallTimeout"
    assert retry.outcome == "returned" and not retry.injected
    assert result.retries == {"payment_status": 1}
```

`result.calls` is the call log: every call in the order it started, with its arguments, outcome and whether a fault was injected. `result.retries` is a heuristic over that log (the same tool called again right after it raised), reported as *inferred*, never as observed.

## Gives up honestly when retries run out

`fail_times(n).then_succeed()` fails the first *n* calls. With two failures and one retry, the agent should say it couldn't confirm, not invent a status:

```python
def test_gives_up_honestly_when_retries_run_out():
    with probe() as p:
        tools = p.wrap(TOOLS)
        p.on("payment_status").fail_times(2).then_succeed()
        result = p.run(lambda q: answer(q, tools), "42")
        result.assert_answer_contains("couldn't confirm")
        result.assert_answer_not_contains("is paid")
```

The scenario plans two failures. If the agent stopped after one call, the scenario would be only *partially exercised* and the gate would fail the test, so a test can't pass because the agent skipped its retry.

## Counting calls

!!! note "Since 1.5"
    `assert_called_times`, `assert_called_at_most`, `assert_call_order` and `assert_called_with` need AgentProbe 1.5 or later.

```python
def test_retries_exactly_once():
    with probe() as p:
        tools = p.wrap(TOOLS)
        p.on("payment_status").fail_first(1)
        result = p.run(lambda q: answer(q, tools), "42")
        result.assert_called_times("payment_status", 2)
        result.assert_called_with("payment_status", "42")
```

These check which calls were made, not whether they succeeded, and each assertion can match a different call. See [Assertions and results](../reference/assertions.md).
