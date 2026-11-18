# Case study: an unhandled tool exception in a LangGraph tutorial agent

**Target:** the email assistant in [langchain-ai/agents-from-scratch](https://github.com/langchain-ai/agents-from-scratch)
(commit 603fc7a), a LangGraph tutorial agent that triages email, checks a calendar, schedules
meetings and writes replies.
**Result:** tool exceptions escape the assistant's tool node and end the run. Reported upstream as
[agents-from-scratch#70](https://github.com/langchain-ai/agents-from-scratch/issues/70) with a
model-free reproduction; a fix with regression tests is proposed in
[agents-from-scratch#71](https://github.com/langchain-ai/agents-from-scratch/pull/71) (not yet reviewed by the maintainers).
**Time:** about an hour from cloning the repo to a confirmed finding.

## What we did

1. **Ran it as-is.** We imported the assistant's own graph, prompts and tools, and swapped in a local
   Ollama model (Qwen 2.5 7B) so no API key or paid inference was needed. The harness replaces three
   module globals; no upstream file is edited. See [`examples/third_party_agent`](../../examples/third_party_agent/).
2. **Wrapped its tools with AgentProbe.** `p.wrap(...)` worked on their `@tool` functions and on
   their pydantic `@tool` classes without changes.
3. **Broke one tool at a time,** three runs per scenario, on the dataset's own meeting-request email.

| Scenario | AgentProbe rule | Observed (3 of 3 runs) |
|---|---|---|
| No fault | none | calendar check, schedule, reply, done |
| Calendar check times out once | `fail_first(1, with_="timeout")` | the run ends with the timeout; the model never sees it |
| Scheduling always fails | `always_fail(with_="api_error")` | the run ends with HTTP 503 after the calendar check |
| Scheduling returns an error message | `returns("Could not schedule ...", times=None)` | no crash; the agent proposes times instead of claiming a booking |

## The finding

The call log made the first two scenarios unambiguous: one tool call, the injected fault, and the
run's exception, with nothing after it. Reading the code explained why: the assistant's
`tool_node` calls `tool.invoke(...)` with no exception handling, so a timeout or an API error ends
the whole run instead of reaching the model, which could otherwise retry or explain.

We then confirmed it without a model and without AgentProbe: calling `tool_node` directly with a tool
that raises propagates the exception. That 15-line reproduction is what went into the upstream
issue, with setup steps checked against the repository's own lock file.

The other three graphs in the repository invoke tools the same way; we inspected them but ran only
the basic graph, and the issue says so. In the Gmail graph, the registered Gmail tools catch their
own errors and return a message, so the gap matters most for tools that raise.

## Before and after the proposed fix

We then ran the example's tests against the original code and against the proposed fix, three runs
each on the same local model ([validation record](../../examples/third_party_agent/results/validation-2026-11-18.md)).
The tests inject `TimeoutError` and LangChain's `ToolException`, the exception types the fix handles:
AgentProbe's own fault classes don't subclass them, so code that handles real timeouts wouldn't
catch them (tracked in AgentProbe [#114](https://github.com/chhayankjain/agentprobe-toolkit/issues/114)).

| Scenario | Original | With the fix (#71) | With #71 and open PR #65 |
|---|---|---|---|
| Calendar times out once | run ends with the timeout | recovers | recovers |
| Scheduling always fails | run ends with the error | `KeyError: None` | recovers |

The fix did what it should, and that exposed a second, already-known bug: once the error reaches the
model, the local model sometimes answers without a tool call, and the graph's router returns `None`
(upstream #61, fixed by open PR #65). With both changes, every scenario passes. Fault tests are good
at finding exactly this kind of interaction, because they exercise the paths that fixes open up.

## Also observed (not reported)

In the third scenario, Qwen 2.5 called the reply tool twice with identical arguments in all three
runs (once without the fault). With a real send tool that would be a duplicate email. It may be a
property of the small local model, so it stays unreported until it is checked on the repository's
own model.

## What it means for your agent

- **Decide what a tool failure should look like to your model.** Returning the error as a tool
  message lets the model retry or explain; letting it escape ends the run. Handle the failures you
  expect (timeouts, connection errors, tool errors) and let programming errors surface.
- **Test the failure paths, not only the happy path.** All of this repository's own tests exercise
  successful tool calls; none injects a failing one.
- **Check behaviour on the call log.** The agent's final message here is a `Done` tool call, so
  there is no answer text to assert on. The call log showed exactly what happened.

## What it taught AgentProbe

- Many agents build their tools at import time. Swapping module globals works, and we've documented
  the pattern in the example; a first-class helper is a candidate for v1.5.
- Graphs that end on a tool call give `result.answer == ""`; checks need the call log. Better answer
  extraction for such graphs is another v1.5 candidate.
- Injected faults should be catchable the way real failures are: an injected timeout should be a
  `TimeoutError` ([#114](https://github.com/chhayankjain/agentprobe-toolkit/issues/114)). Until then, pass the
  exception your code handles with `with_=`.
