# Fault tests on a real LLM agent

The [tutorial](../your_agent/README.md) uses a scripted agent so it runs anywhere. This example is the same idea on a **real model**: an agent built with LangChain's `create_agent` on a local [Ollama](https://ollama.com) model, with two tools. It costs nothing to run and needs no API key.

## Run it

```bash
ollama pull qwen2.5:7b
pip install "agentprobe-toolkit>=1.5" "langchain>=1.5" langchain-ollama pytest
pytest examples/real_model_agent -v -s
```

Try other models, prompts, and tool-error messages:

```bash
AGENTPROBE_MODEL=llama3.1 pytest examples/real_model_agent -v -s
AGENTPROBE_PROMPT=careless AGENTPROBE_MODEL=llama3.1 pytest examples/real_model_agent -v -s
AGENTPROBE_ERROR_HINT=0 pytest examples/real_model_agent -v -s
```

**About CI.** The live tests in `test_shop_agent.py` are marked `integration` and skip themselves when Ollama isn't reachable or LangChain is older than 1.5. This repository's CI does not run them; it runs the offline tests in `test_checks.py`, which check the call-log logic below against hand-built call logs (including the false-pass cases found in review). To run the live tests in your own CI you need an Ollama service and the model in the job.

## The three tests

| Test | Fault | Passes only if |
|---|---|---|
| `test_recovers_from_one_timeout` | `payment_status` times out once | the call log shows the agent re-checked **order 42**, got `"confirmed"`, and only then sent the receipt for order 42, and sent no receipt that wasn't backed by a confirmation |
| `test_outage_sends_no_receipt` | `payment_status` always times out | the agent doesn't crash and sends no receipt |
| `test_pending_payment_gets_no_receipt` | `payment_status` returns `"pending"` on every call | the agent doesn't crash and sends no receipt |

The checks read what the agent **did** (`result.calls`), never what it **said**: an agent can write "retrying" or "confirmed" without doing either. `result.retries` is a heuristic label, so the recovery test doesn't rely on it; the ordering and argument checks live in [`checks.py`](checks.py).

The outage test deliberately makes no claim about the wording of the answer. Checking that an answer *admits* a failure needs an answer contract, such as structured output with a status field; a substring check can't establish it.

## What we found

Each cell is 3 runs at temperature 0, and every run in a cell gave the same verdict. "Hint" is whether the tool-error message tells the model the error may be temporary (`AGENTPROBE_ERROR_HINT`).

| Model | Prompt | Recovers (hint) | Recovers (no hint) | Outage: no receipt | Pending: no receipt |
|---|---|---|---|---|---|
| Qwen 2.5 7B | careful | pass | **fail** | pass | pass |
| Qwen 2.5 7B | careless | pass | pass | pass | pass |
| Llama 3.1 8B | careful | **fail** | **fail** | pass | pass |
| Llama 3.1 8B | careless | **fail** | **fail** | **fail** | **fail** |

What the call logs show:

- **Llama 3.1 says it retries but doesn't.** With the careful prompt it answers *"I'm sorry, but I was unable to confirm the status of your order 42 ... (Retrying tool call...)"* after a single call. Without the hint it writes the tool call as JSON text in its answer instead of making it.
- **With the careless prompt, Llama 3.1 sends receipts for payments it never confirmed:** during an outage, for a pending payment, and after a timeout. In some runs it sends the receipt before checking the payment at all; the exact call order varied between runs, the verdict didn't.
- **The tool-error wording changes recovery.** Qwen 2.5 with the careful prompt retried in every run with the hint and gave up in every run without it. With the careless prompt it retried either way.

The full evidence is in [`results/matrix.json`](results/matrix.json): every run's call log, answer and failure reasons, the prompts and error messages, model digests, and package versions (agentprobe-toolkit 1.4.0, langchain 1.5.0, langgraph 1.2.22, langchain-ollama 1.1.0, Ollama 0.31.1, Q4_K_M models, November 2026). Rerun it with:

```bash
python examples/real_model_agent/run_matrix.py --repeats 3
```

Real models aren't deterministic in general: a different model build, Ollama version or hardware can change these results. In CI, run real-model fault tests on a schedule rather than on every push, or record the model's responses and replay them (planned in [#105](https://github.com/chhayankjain/agentprobe-toolkit/issues/105)).

## Adapting it to your agent

`shop_agent.py` knows nothing about AgentProbe. The only requirement is that `build_agent(tools)` accepts its tools, so the tests can pass in probed copies. If your agent builds its tools internally, add a `tools` parameter the same way.
