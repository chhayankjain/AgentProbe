# Fault tests on a real LLM agent

The [tutorial](../your_agent/README.md) uses a scripted agent so it runs anywhere. This example is the same idea on a **real model**: an agent built with LangChain's `create_agent` on a local [Ollama](https://ollama.com) model, with two tools. It costs nothing to run and needs no API key.

## Run it

```bash
ollama pull qwen2.5:7b
pip install "agentprobe-toolkit>=1.4" "langchain>=1.5" langchain-ollama pytest
pytest examples/real_model_agent -v -s
```

The tests are skipped automatically when Ollama isn't running, so they never break CI. Each takes a few seconds on a laptop.

Try other models and prompts:

```bash
AGENTPROBE_MODEL=llama3.1 pytest examples/real_model_agent -v -s
AGENTPROBE_PROMPT=careless AGENTPROBE_MODEL=llama3.1 pytest examples/real_model_agent -v -s
```

## The three tests

| Test | Fault | The agent must |
|---|---|---|
| `test_recovers_from_one_timeout` | `payment_status` times out once | actually retry, then send the receipt |
| `test_outage_is_admitted_and_no_receipt_is_sent` | `payment_status` always times out | not crash, not send a receipt, not claim the payment is confirmed |
| `test_pending_payment_gets_no_receipt` | `payment_status` returns `"pending"` | not send a receipt |

## What we found

Each cell was run three times at temperature 0; every run gave the same result (Ollama, Q4_K_M models, November 2026).

| Model | Prompt | Recovers from one timeout | Outage | Pending payment |
|---|---|---|---|---|
| Qwen 2.5 7B | careful | pass | pass | pass |
| Qwen 2.5 7B | careless | pass | pass | pass |
| Llama 3.1 8B | careful | **fail** | pass | pass |
| Llama 3.1 8B | careless | **fail** | **fail** | **fail** |

Every failure is a real agent problem, visible in the call log AgentProbe prints:

- **Llama 3.1 says it retries but doesn't.** After one timeout it answers *"I was unable to confirm the status of your order 42 ... (Retrying tool call...)"* and stops. The call log shows a single call. That's why the recovery test checks `result.retries`, not the wording of the answer.
- **With a careless prompt, Llama 3.1 sends receipts for payments it never confirmed**, both during an outage and for a pending payment.

Two lessons that apply to any agent:

1. **How you report tool errors to the model changes whether it recovers.** `shop_agent.report_tool_error` tells the model the error may be temporary. Without that hint, Qwen 2.5 also gave up after the first timeout.
2. **Real models aren't deterministic.** Temperature 0 made these results repeatable on one machine, but a different model version or hardware can change them. In CI, run real-model fault tests on a schedule rather than on every push, or record the model's responses and replay them (planned in [#105](https://github.com/chhayankjain/agentprobe-toolkit/issues/105)).

## Adapting it to your agent

`shop_agent.py` knows nothing about AgentProbe. The only requirement is that `build_agent(tools)` accepts its tools, so the tests can pass in probed copies. If your agent builds its tools internally, add a `tools` parameter the same way.
