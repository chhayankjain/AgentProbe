---
description: Test how an LLM agent handles stale, incomplete or wrong tool results by substituting tool return values in pytest.
---

# Test wrong or stale tool data

Not every failure raises. A tool can return a status the agent doesn't expect, an empty list or yesterday's price. `returns(value)` substitutes the value a call returns, without running the tool, so you can test what the agent does with data that is wrong but well-formed.

## An unexpected status

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

## Stale but valid data

Stale data looks valid: the right fields, plausible values, just old. The only defence is a freshness check, and `returns` lets you test that the agent has one:

```python
from datetime import UTC, datetime, timedelta


def fx_rate(pair: str) -> dict:
    return {"pair": pair, "rate": 1.08, "as_of": datetime.now(UTC).isoformat()}


def quote(amount: float, tools) -> str:
    data = tools["fx_rate"]("EUR/USD")
    age = datetime.now(UTC) - datetime.fromisoformat(data["as_of"])
    if age > timedelta(minutes=15):
        return "The exchange rate is out of date; I can't quote a price right now."
    return f"{amount * data['rate']:.2f} USD"


def test_refuses_to_quote_from_a_day_old_rate():
    day_old = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    with probe() as p:
        tools = p.wrap({"fx_rate": fx_rate})
        p.on("fx_rate").returns({"pair": "EUR/USD", "rate": 1.31, "as_of": day_old})
        result = p.run(lambda a: quote(a, tools), 100)
        result.assert_answer_contains("out of date")
        result.assert_answer_not_contains("131.00")
```

## Combine with failures

Rules for one tool apply in the order you declare them, and each counts only the calls that reach it:

```python
def test_recovers_from_a_timeout_then_an_unexpected_status():
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

Call 1 times out, call 2 returns the unexpected status, call 3 reaches the real tool.

## `returns` versus `"malformed_output"`

- `returns(value)` hands the agent a **value**. Use it for wrong, stale, empty or unparseable data, for example `returns("<html>502 Bad Gateway</html>")` for a JSON tool.
- `with_="malformed_output"` **raises** `ToolCallMalformedOutput`. Use it when your tool wrapper itself raises on bad output and you want to test the agent's handling of that exception.
