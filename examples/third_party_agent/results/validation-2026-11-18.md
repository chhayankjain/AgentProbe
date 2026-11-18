# Before/after validation (2026-11-18)

Separate from `evidence_qwen.json` (the original hunt, which used AgentProbe's own fault classes and is
kept unchanged). This run uses the current tests, which inject the exception types the proposed fix
handles: `TimeoutError` for the calendar timeout and LangChain's `ToolException` for the scheduling
outage.

**Environment:** Python 3.13.13; agentprobe-toolkit 1.4.0, langchain 1.5.1, langchain-core 1.8.3,
langgraph 1.2.22, langchain-ollama 1.1.0, pytest 9.2.0; Ollama 0.31.1, qwen2.5:7b (digest 845dbda0ea48,
Q4_K_M), temperature 0. Upstream code imported from each checkout; nothing installed from it.

**Checkouts**
- original: langchain-ai/agents-from-scratch `603fc7a`
- patched: `603fc7a` + proposed fix [#71](https://github.com/langchain-ai/agents-from-scratch/pull/71) (`4bbd6da`)
- patched + #65: the above merged with open PR [#65](https://github.com/langchain-ai/agents-from-scratch/pull/65) (fix for #61, missing tool calls); merges cleanly

**Command:** `AGENTS_FROM_SCRATCH_DIR=<checkout> pytest examples/third_party_agent -v`

"pass" means the run completed without an unhandled exception after the injected fault fired. The
tests don't check whether the meeting was rescheduled, the reply was truthful, or actions were
duplicated.

| Test | original | patched (#71) | patched + #65 |
|---|---|---|---|
| tool node reports tool errors (no model) | xfail (known bug) | pass | pass |
| survives one calendar timeout | xfail, 3/3 runs | pass, 3/3 | pass, 3/3 |
| survives a scheduling outage | xfail, 3/3 runs | **fail, 3/3**: `KeyError: None` | pass, 3/3 |
| failed scheduling still gets a reply | pass | pass | pass |

**Why the outage test fails with #71 alone.** With the tool error returned to the model, the local model
answers without a tool call. `should_continue` then returns `None`, and LangGraph raises `KeyError: None`.
That is the separate bug #61, fixed by open PR #65; with both changes all four tests pass. This was seen
with qwen2.5:7b through Ollama; whether the repository's default model (gpt-4.1 with `tool_choice="any"`)
also reaches this path was not tested.

**Harness isolation:** entering and leaving `upstream()` restores `OPENAI_API_KEY`/`OPENAI_BASE_URL`,
`sys.path`, and removes the imported `email_assistant` modules (checked: all restored, none left).
