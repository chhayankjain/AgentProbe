# Fault tests on someone else's agent

This example runs AgentProbe against an agent we don't own: the email assistant in
[langchain-ai/agents-from-scratch](https://github.com/langchain-ai/agents-from-scratch), a LangGraph
tutorial agent with calendar and email tools. Nothing in that repository is edited. The full story
is in the [case study](../../docs/case-studies/agents-from-scratch.md).

## Run it

Use Python 3.11 to 3.13 (the upstream project's supported range) and a local Ollama model.

```bash
git clone https://github.com/langchain-ai/agents-from-scratch && git -C agents-from-scratch checkout 603fc7a
pip install -e ./agents-from-scratch "agentprobe-toolkit>=1.4" langchain-ollama pytest
ollama pull qwen2.5:7b

AGENTS_FROM_SCRATCH_DIR=./agents-from-scratch pytest examples/third_party_agent -v -s
```

Without `AGENTS_FROM_SCRATCH_DIR` the tests skip, and the three model-backed tests skip when Ollama
isn't running. No OpenAI key is needed and no request leaves your machine.

## What the tests show

| Test | Fault | Today |
|---|---|---|
| `test_tool_node_reports_tool_errors_to_the_model` | a tool raises (no model involved) | **xfail**: the exception escapes the tool node |
| `test_survives_one_calendar_timeout` | the calendar check times out once | **xfail**: the run ends with the timeout |
| `test_survives_a_scheduling_outage` | scheduling always fails with HTTP 503 | **xfail**: the run ends with the error |
| `test_failed_scheduling_still_gets_a_reply` | scheduling returns an error message instead of raising | passes: the agent replies |

The expected failures are reported upstream as
[agents-from-scratch#70](https://github.com/langchain-ai/agents-from-scratch/issues/70), with a fix
proposed in [#71](https://github.com/langchain-ai/agents-from-scratch/pull/71). They are
marked `xfail(strict=True)`, so they turn red here as soon as the upstream bug is fixed.

## How the harness works

The assistant builds its model and tools when its module is imported, and has no parameter for
passing tools in. [`harness.py`](harness.py) imports it with a placeholder OpenAI key and a dead
OpenAI address, then swaps three module globals for a local model and AgentProbe-wrapped copies of
the same tools. Their prompts and graph run unchanged. Use the same pattern for any agent that
builds its tools at import time; if you own the agent, add a `tools` parameter instead.

`results/evidence_qwen.json` holds the call logs from the original runs (3 per scenario).
