---
description: Make injected tool faults look like real failures. Raise TimeoutError, ConnectionError, LangChain ToolException or your HTTP client's errors from AgentProbe fault tests.
---

# Inject the exceptions your code handles

Code that handles real failures catches specific types: `TimeoutError`, `ConnectionError`, an HTTP client's errors, LangChain's `ToolException`. If a fault test raises a different type, the handler never runs and the test reports a crash the real failure wouldn't cause. Pass the exception your code expects with `with_=`.

## Any exception class or instance

```python
from langchain_core.tools import ToolException

from agentprobe.testing import probe


def fetch_rates(currency: str) -> float:
    return 1.08


def convert(amount: float, tools) -> str:
    try:
        rate = tools["fetch_rates"]("EUR")
    except (ConnectionError, ToolException):
        return "Exchange rates are unavailable right now."
    return f"{amount * rate:.2f} USD"


def test_connection_refused():
    with probe() as p:
        tools = p.wrap({"fetch_rates": fetch_rates})
        p.on("fetch_rates").fail_first(1, with_=ConnectionError("connection refused"))
        result = p.run(lambda a: convert(a, tools), 100)
        result.assert_answer_contains("unavailable")


def test_tool_exception_class():
    with probe() as p:
        tools = p.wrap({"fetch_rates": fetch_rates})
        p.on("fetch_rates").always_fail(with_=ToolException)
        result = p.run(lambda a: convert(a, tools), 100)
        result.assert_answer_contains("unavailable")
```

Each call raises a fresh exception: a shallow copy of an instance, or a new instance of a class, created with the message "<ClassName> injected by AgentProbe".

## Common choices

| Your code catches | Pass |
|---|---|
| `TimeoutError` | `with_=TimeoutError("timed out")`, or `"timeout"` from the next release |
| `ConnectionError` | `with_=ConnectionError("connection refused")` |
| LangChain `ToolException` | `with_=ToolException("service unavailable")` |
| `httpx` timeouts | `with_=httpx.ReadTimeout("read timed out")` |
| your own error type | `with_=PaymentGatewayError(...)` |

!!! note "Next release"
    From the next release, the built-in `"timeout"` fault is also a `TimeoutError`, so `except TimeoutError` catches it without `with_=` ([#114](https://github.com/chhayankjain/agentprobe-toolkit/issues/114)). The other built-in faults are subclasses of `ToolCallError` only.

For bad data rather than an exception, use [`returns(...)`](wrong-data.md).
