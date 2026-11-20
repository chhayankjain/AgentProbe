---
description: Reference for AgentProbe fault scenarios - fail_first, fail_times, always_fail, returns, slow, with_probability - and the built-in faults.
---

# Scenarios

A scenario says what happens to calls to one tool. Declare it with `p.on("tool_name")` (or `p.on(wrapped_tool)`) inside a `probe()` block, before the agent runs.

```py
with probe() as p:
    tools = p.wrap(TOOLS)
    p.on("search").fail_first(1)
```

## Methods

| Method | What happens | Default fault |
|---|---|---|
| `fail_first(n=1, *, with_="timeout")` | The first *n* calls raise, later calls run the real tool | `"timeout"` |
| `fail_times(n, *, with_=...).then_succeed()` | The same, spelled out | `"timeout"` |
| `always_fail(*, with_="timeout")` | Every call raises | `"timeout"` |
| `returns(value, *, times=1)` | The first *times* calls return *value* without running the tool; `times=None` means every call | |
| `slow(*, ms, times=1)` | The first *times* calls wait *ms* milliseconds, then run the real tool | |
| `with_probability(p, *, with_="timeout", seed=None)` | Each call raises with probability *p*; reproducible for a fixed *seed* | `"timeout"` |

Every method also takes `required=True`. A required scenario must be **exercised** or the gate fails the test; pass `required=False` for a scenario that may legitimately not fire.

## Faults

`with_=` takes a built-in fault name, an exception class or an exception instance.

| Name | Raises |
|---|---|
| `"timeout"` | `ToolCallTimeout` |
| `"rate_limit"` | `ToolCallRateLimited` (simulates HTTP 429; `retry_after=1.0`) |
| `"api_error"` | `ToolCallAPIError` (`status_code=503`) |
| `"malformed_output"` | `ToolCallMalformedOutput` |

All are subclasses of `ToolCallError`, importable from `agentprobe.testing` (from the next release; in 1.4.0, from `agentprobe.injector.tool_failure`). From the next release `ToolCallTimeout` is also a `TimeoutError`. See [Inject the exceptions your code handles](../guides/real-exceptions.md) for passing your own types.

## Order and counting

Rules for the same tool form a sequence. Each call is offered to the rules in declaration order, and each rule counts only the calls that reach it:

```py
p.on("search").fail_first(1).returns("stale", times=2)
# call 1: timeout, calls 2-3: "stale", call 4 onward: the real tool
```

A rule that applies to every call (`always_fail()`, or `times=None`) must be the last rule for its tool, because a rule after it could never run; declaring one raises `ProbeError`.

## Evidence

For every scenario the probe records whether the tool was reached, how many faults were planned and how many were injected, and a status:

| Status | Meaning |
|---|---|
| `exercised` | Every planned fault was injected (open-ended rules: at least one) |
| `partially_exercised` | Some, but not all, planned faults were injected |
| `not_exercised` | No fault was injected |

Only `exercised` satisfies a required scenario. Read it from `result.scenarios`.
