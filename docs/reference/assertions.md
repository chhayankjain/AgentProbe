---
description: Reference for AgentProbe assertions and ProbeResult - answer checks, crash and deadline checks, tool-call checks, the call log, the gate and JSON reports.
---

# Assertions and results

`p.run(agent, input)` returns a `ProbeResult`. Its assertions raise `AssertionError` immediately **and** are recorded, so the gate at the end of the `with` block still fails the test if you catch one. Every assertion returns the result, so you can chain them.

## Outcome assertions

| Assertion | Passes when |
|---|---|
| `assert_answer_contains(text, *, case_sensitive=False)` | The agent produced an answer containing *text* |
| `assert_answer_not_contains(text, *, case_sensitive=False)` | The agent produced an answer **without** *text*. A crash or `None` never passes |
| `assert_no_crash()` | No exception escaped the agent |
| `assert_within(seconds)` | The run finished in time |
| `assert_scenarios_exercised()` | Every required scenario fired (also enforced when the block ends) |

## Tool-call assertions

| Assertion | Passes when | Available |
|---|---|---|
| `assert_called(tool)` | The tool was called at least once | 1.4 |
| `assert_not_called(tool)` | The tool was never called | 1.4 |
| `assert_called_with(tool, *args, **kwargs)` | **Any** call had these arguments; arguments you leave out aren't checked | next release |
| `assert_call_order([...])` | The tools were called in this order, other calls allowed between | next release |
| `assert_called_times(tool, n)` | Exactly *n* calls started | next release |
| `assert_called_at_most(tool, n)` | At most *n* calls started | next release |

Calls still in flight count. Calls that start after the probe closed never count.

`assert_called_with` differs from `unittest.mock`: it matches any call, not only the last. When the tool's signature is known, `tool("42")` and `tool(order_id="42")` match each other. Values are compared exactly against a copy taken when the call started; only built-in values (`None`, `bool`, numbers, `str`, bytes, dates and times, `Decimal`, `UUID`, paths, enum members, and lists, tuples, dicts and sets of these) can be compared. Any other argument makes the assertion *unassessed*, never passed. Details in [How probing works](../instrumentation.md#tool-call-assertions-compare-exact-copies).

Every assertion except `assert_called_with` takes `required=True`; a non-required assertion is recorded but doesn't fail the gate.

## Pass, fail and unassessed

An assertion that can't be decided is **unassessed**, never passed: for example an answer check when the agent is still running at its deadline, or was cancelled. A required unassessed assertion fails the gate.

> A probe passes only when every required scenario was exercised and every required assertion passed. Unassessed results, errors and missing evidence never pass.

## The result

| Field | What it holds |
|---|---|
| `answer` | The agent's final answer as text, or `None` |
| `raw_output` | What the agent returned, unchanged |
| `exception` | The exception that escaped the agent, or `None` |
| `elapsed_ms`, `deadline_exceeded`, `still_running`, `cancelled` | Timing and deadline outcome |
| `calls` | The call log: one `CallRecord` per tool call, in the order calls started |
| `scenarios` | One `ScenarioEvidence` per declared scenario |
| `assertions` | Every assertion made, with status and message |
| `retries`, `fallbacks` | **Inferred** from the call log: a tool called again after it raised; a different tool called after one raised |

Each `CallRecord` has the tool name, arguments (`args`, `kwargs`, a JSON-safe snapshot), `outcome` (`"returned"` or `"raised"`), `value` or `exception_type`, `injected` and `fault`, and timing.

## Reports

- `print(result)` or `result.summary()` shows what was **observed**, what was **asserted** and what was **inferred**, kept apart.
- `result.to_dict()` is versioned (`schema_version`), JSON-safe output for CI artifacts.
- `result.gate()` returns the gate verdict and the reasons it failed.
