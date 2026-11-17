# Instrumentation contract

How `agentprobe.testing.probe()` touches your tools, and what it guarantees. Tracked in issue #95.

## Your tools are never modified

`p.wrap(tools)` returns **new objects**:

| You pass | You get back |
|---|---|
| a plain function (sync or async) | a new function with the same name, signature and docstring (`__wrapped__` points to yours) |
| a LangChain `BaseTool` | a shallow copy (`model_copy()`) whose `_run` (and `_arun`, if the tool has its own async implementation) is wrapped |
| a list, tuple or `{name: tool}` dict | the same container type, with each tool wrapped |

The schema a model sees for a wrapped LangChain tool is identical to the original. Because nothing is patched in place, there is nothing to restore. Two probes wrapping the same original tool, at the same time or one after another, cannot interfere with each other.

**What the tests check after every probe:** the original's methods are the same objects as before, its attributes (`vars()`) are unchanged, and no probe state is reachable from it.

Wrapping in place (needed to probe an already-compiled graph, #64) will come later. It will reject an overlapping probe on the same tool object instead of interleaving.

## State

Each probe owns its rules, its call log and its per-tool counters, behind a lock. Nothing is module-global. Concurrent probes in different threads are independent.

Rules for the same tool form a sequence. Each call is offered to the rules in declaration order, and each rule counts only the calls that reach it:

```python
p.on("search").fail_first(1).returns("stale", times=2)
# call 1: timeout, calls 2-3: "stale", call 4 onward: real result
```

**Validation at declaration.** Counts (`fail_first(n)`, `times=` on `returns` and `slow`) must be positive whole numbers, or `None` where "every call" is allowed. Delays must be finite and non-negative; probabilities between 0 and 1. A rule that applies to every call (`always_fail()`, or `times=None`) must be the last rule for its tool: a rule declared after it could never run, so declaring one raises `ProbeError`. Invalid scenarios fail when you write them, never by silently counting as exercised.

## Lifecycle and cleanup

The probe is active inside its `with` block (or `async with`). Leaving the block closes it, whether the block ended normally, raised an exception, or was cancelled (`asyncio.CancelledError`).

- **Normal exit:** the gate runs. The test fails with `ProbeGateError` unless every required scenario was exercised and every required assertion passed.
- **Exit by exception:** the probe closes and your exception propagates unchanged. The gate does not mask it.
- A probe cannot be reused once it has closed. Use a new probe per test.

## Every call is recorded when it starts

Each tool call is added to the log the moment it begins, with `status="pending"`, and completed when it returns or raises. Invocation order (`index`, `started_seq`) is allocated at entry, so concurrent calls never share an index, and completion order is kept separately (`completed_seq`). A call still in flight at a deadline is visible: `assert_not_called` fails on it, `assert_called` passes, and the summary shows it as `IN FLIGHT`.

Nested tools are each recorded. If a wrapped tool calls another wrapped tool, both calls appear and both get their faults. Only a tool delegating to *itself* (a LangChain tool's async method calling its own sync method) is counted once.

## Agents that are still running

`p.run(agent, input, timeout=seconds)` runs the agent in a worker thread. Python cannot force-stop a thread, so an agent can outlive its deadline. When that happens:

1. The result has `deadline_exceeded=True` and `still_running=True`, and no final answer. The result is a frozen snapshot: nothing that happens later changes it.
2. Assertions that need an answer are `unassessed`, as is `assert_not_called` for a tool not called *yet*. A required unassessed assertion fails the gate.
3. The probe closes immediately. A call that was already in flight finishes in the probe's live log with `completed_after_close=True`. A call that *starts* later passes straight through **without a fault** and is recorded in `p.late_calls` with `after_probe_end=True`.
4. Neither kind of late activity counts as scenario evidence or appears in another probe's log.

The deadline is measured by waiting on the worker, not by the type of exception: if your agent raises its own `TimeoutError` before the deadline, that is the agent's exception and `deadline_exceeded` stays `False`.

Without `timeout`, the agent runs in the calling thread, and `elapsed_ms` is still recorded.

## Async agents and async tests

Use `await p.arun(agent, input, timeout=...)` from async code. An async agent runs in the current event loop and, unlike a thread, **can be stopped**: at the deadline it is cancelled, the result has `deadline_exceeded=True` and `cancelled=True`, and answer assertions are `unassessed`. A sync agent passed to `arun` runs in a worker thread and behaves like `run`.

`p.run()` refuses an async agent when an event loop is already running in the thread (as in an async test) and tells you to use `arun`, before creating any coroutine.

## Reports

`result.summary()` prints what was observed, asserted and inferred. `result.to_dict()` is always JSON-safe (schema version `1.1`): JSON values pass through; datetimes, decimals, UUIDs, paths and enums become strings; bytes become `{"type": "bytes", "length", "preview"}`; anything else becomes `{"type", "repr"}`. Your objects are never modified.

## Evidence

For every scenario the probe records, from its own call log only:

| Field | Meaning |
|---|---|
| `boundary_reached` | the tool was called while the probe was active |
| `planned` | faults the rule intends to apply (`None` for open-ended rules) |
| `injected` | faults actually applied |
| `status` | `exercised`, `partially_exercised` or `not_exercised` |
| `required` | `True` by default; opt out per rule with `required=False` |

Only `exercised` satisfies a required scenario. An agent that skips the tool, or answers from a cache, cannot pass a fault test.

## Gate rule

> A probe passes only when every required scenario was exercised and every required assertion passed. Unassessed results, errors and missing evidence never pass.

Answer assertions require an actual answer. If the agent crashed, returned `None`, is still running, or was cancelled, `assert_answer_contains` **and** `assert_answer_not_contains` both fail or are unassessed: "no answer" is never treated as "an answer without the claim".

Assertions raise immediately, and are also recorded. Catching an `AssertionError` inside the block does not make the gate pass.
