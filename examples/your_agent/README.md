# Tutorial: add a fault test to your existing agent

Ten minutes, no model and no API key. You'll take an ordinary agent, break one of its tools on purpose, and check that the agent still gives an honest answer. Then you'll watch the same test catch a buggy version of the agent, and run it in CI.

## 1. The agent

[`agent.py`](agent.py) answers "did my order go through?" with two tools: `payment_status` and `send_receipt`. It stands in for your agent. The only thing AgentProbe needs is that the agent receives its tools as arguments, so a test can hand it probed copies.

```python
def answer(question, tools=TOOLS, *, broken=False): ...
```

## 2. Install

```bash
pip install agentprobe-toolkit pytest
```

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
