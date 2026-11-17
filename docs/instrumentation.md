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

## Lifecycle and cleanup

The probe is active inside its `with` block (or `async with`). Leaving the block closes it, whether the block ended normally, raised an exception, or was cancelled (`asyncio.CancelledError`).

- **Normal exit:** the gate runs. The test fails with `ProbeGateError` unless every required scenario was exercised and every required assertion passed.
- **Exit by exception:** the probe closes and your exception propagates unchanged. The gate does not mask it.
- A probe cannot be reused once it has closed. Use a new probe per test.

## Agents that are still running

`p.run(agent, input, timeout=seconds)` runs the agent in a worker thread. Python cannot force-stop a thread, so an agent can outlive its deadline. When that happens:

1. The result has `deadline_exceeded=True` and `still_running=True`, and no final answer.
2. Assertions that need an answer are `unassessed`. A required unassessed assertion fails the gate.
3. The probe closes immediately. Any later tool call through its wrappers **passes straight through without a fault** and is recorded in `p.late_calls` with `after_probe_end=True`.
4. Late calls never count as scenario evidence, and never appear in another probe's log.

Without `timeout`, the agent runs in the calling thread, and the result's `elapsed_ms` is still recorded.

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

Assertions raise immediately, and are also recorded. Catching an `AssertionError` inside the block does not make the gate pass.
