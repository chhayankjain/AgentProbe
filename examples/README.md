# Examples

| File | What it shows | Needs |
|---|---|---|
| `quickstart.py` | Inject seeded faults into a LangGraph agent, run a 10-run benchmark, print metrics and the span-linked injection log | nothing: uses the scripted model |
| [`your_agent/`](your_agent/README.md) | Test an existing agent's tools with deterministic scenarios, assertions, and a pytest CI workflow | pytest; no model or API key |
| [`real_model_agent/`](real_model_agent/README.md) | The same tests on a real LLM agent (LangChain `create_agent` on local Ollama), with results across models and prompts | Ollama; `langchain>=1.5`; no API key |

Run the benchmark example from a clone with `python examples/quickstart.py`, or run the packaged demo with `python -m agentprobe.demo` (`--fault`, `--p`, `--runs`, `--seed` flags). The demo also grades the final calculator answer. For tests of your own agent, start with the [`your_agent` tutorial](your_agent/README.md).

The experiment scripts behind the paper live in `../experiments/`.
