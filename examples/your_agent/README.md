# Tutorial: add a fault test to your existing agent

Ten minutes, no model and no API key. You'll take an ordinary agent, break one of its tools on purpose, and check that the agent still gives an honest answer. Then you'll watch the same test catch a buggy version of the agent, and run it in CI.

## 1. The agent

[`agent.py`](agent.py) answers "did my order go through?" with two tools: `payment_status` and `send_receipt`. It stands in for your agent. The only thing AgentProbe needs is that the agent receives its tools as arguments, so a test can hand it probed copies.

```python
def answer(question, tools=TOOLS, *, broken=False): ...
```

## 2. Install

```bash
pip install "agentprobe-toolkit>=1.5" pytest
```

This tutorial needs AgentProbe 1.5 or later. No model or API key is needed for this tutorial.

## 3. Write the fault test

```python
from agentprobe.testing import probe
from agent import TOOLS, answer


def test_admits_failure_when_payment_service_is_down():
    with probe() as p:
        tools = p.wrap(TOOLS)  # probed copies; TOOLS is untouched
        p.on("payment_status").always_fail(with_="timeout")  # the fault
        result = p.run(lambda q: answer(q, tools), "Did order 42 go through?")
        result.assert_no_crash()
        result.assert_answer_not_contains("is confirmed")  # never claim unverified success
        result.assert_not_called("send_receipt")
```

Run it:

```bash
pytest examples/your_agent
```

When the `with` block ends, AgentProbe also checks that the fault **actually fired**. If your agent never called `payment_status`, maybe because it answered from a cache, the test fails with "scenario not exercised" instead of passing quietly.

## 4. Watch it catch a real bug

The broken agent swallows the payment error and reports success anyway:

```python
except Exception:
    status = "confirmed"   # BUG
```

Point the same test at `answer(q, tools, broken=True)` and it fails:

```
AssertionError: answer does not contain 'is confirmed': answer was 'Order 42 is confirmed and paid; receipt sent for order 42.'

Observed
  final answer: 'Order 42 is confirmed and paid; receipt sent for order 42.'
  1. payment_status('42') raised ToolCallTimeout: Tool call timed out (injected by AgentProbe)  [injected: ...]
  2. send_receipt('42') returned 'receipt sent for order 42'
  scenarios:
    [exercised] payment_status: fail every call with timeout (required; ...)
```

The report shows what happened in order: the payment check timed out, the agent sent a receipt anyway, and claimed success. [`test_agent.py`](test_agent.py) keeps this as `test_broken_agent_is_caught`.

## 5. More scenarios

```python
p.on("payment_status").fail_first(1, with_="timeout")  # fail once, then work: does it retry?
p.on("payment_status").fail_times(2).then_succeed()  # same, spelled out
p.on("payment_status").returns("unknown")  # plausible but wrong data
p.on("payment_status").slow(ms=3000)  # slow but working
p.on("payment_status").always_fail(
    with_="rate_limit"
)  # also "api_error", "malformed_output", or any exception
```

Useful checks on the result: `assert_answer_contains`, `assert_answer_not_contains`, `assert_no_crash`, `assert_within(seconds)`, `assert_called`, `assert_not_called`. `print(result)` shows what was observed, what you asserted, and what was inferred (retries and fallbacks).

### Check the tool calls themselves

This test checks which tools the agent invoked after a transient failure.

```python
with probe() as p:
    tools = p.wrap(TOOLS)
    p.on("payment_status").fail_first(1)
    result = p.run(lambda q: answer(q, tools), "Did order 42 go through?")
    result.assert_called_with("payment_status", "42")
    result.assert_call_order(["payment_status", "payment_status", "send_receipt"])
    result.assert_called_times("payment_status", 2)
    result.assert_called_at_most("send_receipt", 1)
```

- `assert_called_with` passes if **any** call to the tool had these arguments; arguments you leave out aren't checked. This differs from `unittest.mock`, where `assert_called_with` checks only the last call and every argument. `payment_status("42")` and `payment_status(order_id="42")` match each other, including for LangChain tools. Every keyword, including `required`, is matched against the tool call.
- Values are compared exactly (Python's `==`) against a copy taken when the call started, so changing an object later doesn't change the verdict. Only built-in values are supported (`None`, `bool`, numbers, `str`, bytes, dates and times, `Decimal`, `UUID`, paths, enum members, and lists, tuples, dicts and sets of these). If the agent passed anything else, such as a dataclass or a pydantic model, the assertion is `unassessed`, never passed, and using one as the expected value raises `ProbeError`.
- `assert_call_order` checks the order in which calls **started**, and allows other calls in between.
- Calls still in flight count. While the agent is still running, a match that was already seen passes and an exceeded count fails; anything that could still change is `unassessed`.

These assertions check which calls were made, not whether they succeeded, and each one can match a different call. Here, `assert_called_with("payment_status", "42")` is satisfied by the first call, the one that failed. To check that the receipt was sent only after a payment check that *returned*, look at the call log: `[c for c in result.calls if c.tool == "payment_status" and c.outcome == "returned"]`.

## 6. Run it in CI

Copy [`ci.yml`](ci.yml) to `.github/workflows/agent-fault-tests.yml` in your repository. It installs AgentProbe, runs your fault tests, and uploads a JUnit report.

## Using LangChain or LangGraph

`p.wrap()` also accepts LangChain tools and returns copies with the same schema, so you can build your agent from the wrapped tools exactly as usual:

```python
tools = p.wrap([payment_status_tool, send_receipt_tool])
agent = create_react_agent(model, tools)
result = p.run(agent, {"messages": [("user", "Did order 42 go through?")]})
```

One thing this surfaces straight away: LangGraph's prebuilt agent doesn't catch arbitrary tool exceptions by default, so a single tool timeout crashes it. `ToolNode(tools, handle_tool_errors=True)` turns the error into a message the model can react to.

## How it works

[`docs/instrumentation.md`](../../docs/instrumentation.md) explains what AgentProbe guarantees about your tools, its state, cleanup, and agents that are still running when a deadline passes.
