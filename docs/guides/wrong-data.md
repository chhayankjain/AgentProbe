---
description: Test how an LLM agent handles stale, incomplete or wrong tool results by substituting tool return values in pytest.
---

# Test wrong or stale tool data

Not every failure raises. A tool can return yesterday's price, an empty list or a status the agent doesn't expect. `returns(value)` substitutes the value a call returns, without running the tool, so you can test what the agent does with data that is wrong but well-formed.

## A stale status

```python
from agentprobe.testing import probe


def order_status(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped"}


def answer(order_id: str, tools) -> str:
    status = tools["order_status"](order_id).get("status")
    if status not in {"pending", "shipped", "delivered"}:
        return f"I couldn't determine the status of order {order_id}."
    return f"Order {order_id} is {status}."


def test_does_not_repeat_an_unknown_status():
    with probe() as p:
        tools = p.wrap({"order_status": order_status})
        p.on("order_status").returns({"order_id": "42", "status": "UNKNOWN_7"})
        result = p.run(lambda q: answer(q, tools), "42")
        result.assert_answer_contains("couldn't determine")
        result.assert_answer_not_contains("UNKNOWN_7")
```

By default `returns` applies to the first call only (`times=1`). Use `times=n` for the first *n* calls or `times=None` for every call.

## Combine with failures

Rules for one tool apply in the order you declare them, and each counts only the calls that reach it:

```python
def test_recovers_from_a_timeout_then_stale_data():
    with probe() as p:
        tools = p.wrap({"order_status": order_status})
        p.on("order_status").fail_first(1).returns({"status": "?"}, times=1)
        first = p.run(lambda q: _safe(answer, q, tools), "42")
        second = p.run(lambda q: answer(q, tools), "42")
        third = p.run(lambda q: answer(q, tools), "42")
    assert first.answer == "error"
    assert "couldn't determine" in second.answer
    assert third.answer == "Order 42 is shipped."


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return "error"
```

Call 1 times out, call 2 returns the stale value, call 3 reaches the real tool.

## `returns` versus `"malformed_output"`

- `returns(value)` hands the agent a **value**. Use it for wrong, stale, empty or unparseable data, for example `returns("<html>502 Bad Gateway</html>")` for a JSON tool.
- `with_="malformed_output"` **raises** `ToolCallMalformedOutput`. Use it when your tool wrapper itself raises on bad output and you want to test the agent's handling of that exception.
