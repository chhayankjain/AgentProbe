# Fault tests on someone else's agent

This example runs AgentProbe against an agent we don't own: the email assistant in
[langchain-ai/agents-from-scratch](https://github.com/langchain-ai/agents-from-scratch), a LangGraph
tutorial agent with calendar and email tools. Nothing in that repository is edited. The full story
is in the [case study](../../docs/case-studies/agents-from-scratch.md).

## Run it

Use Python 3.11 to 3.13 (the upstream project's supported range) and a local Ollama model.

```bash
git clone https://github.com/langchain-ai/agents-from-scratch && git -C agents-from-scratch checkout 603fc7a
pip install -e ./agents-from-scratch "agentprobe-toolkit>=1.5" langchain-ollama pytest
ollama pull qwen2.5:7b

AGENTS_FROM_SCRATCH_DIR=./agents-from-scratch pytest examples/third_party_agent -v -s
```

Without `AGENTS_FROM_SCRATCH_DIR` the tests skip, and the three model-backed tests skip when Ollama
isn't running. No OpenAI key is needed and no request leaves your machine.

## What the tests show

The tests describe the behaviour we'd want: a failing tool shouldn't end the run. They inject the
exception types the proposed upstream fix handles (`TimeoutError`, LangChain's `ToolException`).

| Test | Fault | On `603fc7a` |
|---|---|---|
| `test_tool_node_reports_tool_errors_to_the_model` | a tool raises (no model involved) | xfail: the exception escapes the tool node |
| `test_survives_one_calendar_timeout` | the calendar check raises `TimeoutError` once | xfail: the run ends with the timeout |
| `test_survives_a_scheduling_outage` | scheduling always raises `ToolException` | xfail: the run ends with the error |
| `test_failed_scheduling_still_gets_a_reply` | scheduling returns an error message instead of raising | passes: the agent replies |

A test is marked as an expected failure **only** when the injected fault fired and exactly that
exception escaped, the known bug reported as
[agents-from-scratch#70](https://github.com/langchain-ai/agents-from-scratch/issues/70). Anything else
(the model isn't running, the fault never fired, a different exception) is a real failure.

The commit is pinned, so these results don't change on their own. Point `AGENTS_FROM_SCRATCH_DIR` at a
checkout with a fix and they pass instead of xfailing. Passing means the run completes without an
unhandled exception; the tests don't check what the agent did next. The before/after run against the
proposed fix ([#71](https://github.com/langchain-ai/agents-from-scratch/pull/71)) is in
[`results/validation-2026-11-18.md`](results/validation-2026-11-18.md): with #71 the timeout test
passes, and the outage test then hits a separate known bug (#61) until open PR #65 is applied too;
with both, all four pass.

## How the harness works

The assistant builds its model and tools when its module is imported, and has no parameter for
passing tools in. [`harness.py`](harness.py) imports it with a placeholder OpenAI key and a dead
OpenAI address, then swaps three module globals for a local model and AgentProbe-wrapped copies of
the same tools. Their prompts and graph run unchanged. Use the same pattern for any agent that
builds its tools at import time; if you own the agent, add a `tools` parameter instead.

Both helpers are context managers and undo everything on exit (environment variables, `sys.path`,
the imported upstream modules, the swapped globals), so the example can run alongside other tests.

`results/evidence_qwen.json` holds the call logs from the original hunt (3 runs per scenario, using
AgentProbe's own fault classes). `results/validation-2026-11-18.md` records the before/after run.
