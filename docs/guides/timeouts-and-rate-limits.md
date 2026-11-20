---
description: Simulate tool timeouts, rate limits (HTTP 429), API errors and slow tools in pytest to test how an LLM agent behaves.
---

# Simulate timeouts and rate limits

The built-in faults cover the failures tools hit most: timeouts, rate limits, server errors and slow responses. All of them are deterministic: the same calls fail on every run.

| Fault (`with_=`) | Raises | Also an instance of |
|---|---|---|
| `"timeout"` (default) | `ToolCallTimeout` | `ToolCallError`; `TimeoutError` from the next release |
| `"rate_limit"` | `ToolCallRateLimited` (HTTP 429, `retry_after`) | `ToolCallError` |
| `"api_error"` | `ToolCallAPIError` (HTTP 503) | `ToolCallError` |
| `"malformed_output"` | `ToolCallMalformedOutput` | `ToolCallError` |

All four live in `agentprobe.injector.tool_failure`. To raise the exact exception your client raises instead, see [Inject the exceptions your code handles](real-exceptions.md).

## A rate-limited search tool

```python
from agentprobe.injector.tool_failure import ToolCallRateLimited
from agentprobe.testing import probe


def search(query: str) -> str:
    return f"3 results for {query!r}"


def research_agent(question: str, tools) -> str:
    try:
        return tools["search"](question)
    except ToolCallRateLimited as exc:
        return f"Search is rate limited; try again in {exc.retry_after:.0f}s."


def test_reports_rate_limit_instead_of_guessing():
    with probe() as p:
        tools = p.wrap({"search": search})
        p.on("search").always_fail(with_="rate_limit")
        result = p.run(lambda q: research_agent(q, tools), "fault injection")
        result.assert_no_crash()
        result.assert_answer_contains("rate limited")
        result.assert_answer_not_contains("results for")
```

## A slow tool and a deadline

`slow(ms=...)` delays a call and then runs the real tool. Combine it with `p.run(..., timeout=seconds)` to test a latency budget:

```python
def test_answers_within_budget_despite_a_slow_call():
    with probe() as p:
        tools = p.wrap({"search": search})
        p.on("search").slow(ms=200)
        result = p.run(lambda q: research_agent(q, tools), "slow", timeout=5)
        result.assert_within(2)
        result.assert_answer_contains("results for")
```

If the agent is still running at the deadline, the result says `deadline_exceeded=True` and `still_running=True`, and answer assertions are *unassessed*, which fails the gate. Python can't stop a thread, so AgentProbe never reports a run that outlived its deadline as passed. Async agents (`await p.arun(...)`) are cancelled at the deadline; see [How probing works](../instrumentation.md#agents-that-are-still-running).

## Random faults for experiments

`with_probability(p, seed=...)` injects faults at random, reproducibly for a given seed. Prefer the deterministic scenarios in tests: a test that fails one run in five is hard to trust.
