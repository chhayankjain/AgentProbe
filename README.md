# AgentProbe

[![CI](https://github.com/chhayankjain/agentprobe-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/chhayankjain/agentprobe-toolkit/actions/workflows/ci.yml)
[![CodeQL](https://github.com/chhayankjain/agentprobe-toolkit/actions/workflows/codeql.yml/badge.svg)](https://github.com/chhayankjain/agentprobe-toolkit/actions/workflows/codeql.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/agentprobe-toolkit.svg)](https://pypi.org/project/agentprobe-toolkit/)
[![GitHub release](https://img.shields.io/github/v/release/chhayankjain/agentprobe-toolkit)](https://github.com/chhayankjain/agentprobe-toolkit/releases)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://chhayankjain.github.io/agentprobe-toolkit/)

**Test how your LLM agent handles tool failures, in pytest.**

Your agent's tools will time out, hit rate limits, and return garbage. AgentProbe wraps the tools your agent already uses, breaks them on purpose and deterministically, and checks what your agent did about it. The test fails unless every fault you declared **actually fired**, so an agent that skips the tool or answers from a cache can't look resilient. The examples and the demo run without a model or API key.

[Docs](https://chhayankjain.github.io/agentprobe-toolkit/) · [Install](#install) · [Your first fault test](#your-first-fault-test) · [What a failure looks like](#what-a-failure-looks-like) · [Scenarios](#what-you-can-break) · [Assertions](#what-you-can-check) · [CI](#run-it-in-ci) · [Tutorial](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/examples/your_agent/README.md) · [For researchers](#for-researchers-the-benchmark-and-the-paper)

---

## Install

```bash
pip install "agentprobe-toolkit>=1.4" pytest
```

The testing API on this page (`agentprobe.testing`) needs version 1.4 or later.

You install `agentprobe-toolkit` and import `agentprobe`, the same way `pip install scikit-learn` gives you `import sklearn`. Python 3.11+.

## Your first fault test

Say `answer(question, tools)` is your agent and `TOOLS` maps tool names to the functions it calls. This test checks that when the payment service is down, the agent admits it instead of claiming success:

```python
from agentprobe.testing import probe

from my_agent import TOOLS, answer


def test_admits_failure_when_payment_service_is_down():
    with probe() as p:
        tools = p.wrap(TOOLS)                                   # probed copies; your tools are untouched
        p.on("payment_status").always_fail(with_="timeout")     # break one tool, every call
        result = p.run(lambda q: answer(q, tools), "Did order 42 go through?")
        result.assert_no_crash()
        result.assert_answer_not_contains("is confirmed")       # never claim unverified success
        result.assert_answer_contains("couldn't confirm")       # say so instead
        result.assert_not_called("send_receipt")                # no receipt for an unconfirmed payment
```

Run it with `pytest`. Nothing is mocked except the fault: your agent runs for real, with real tools, and AgentProbe only steps in at the tool boundary.

## What a failure looks like

Point the same test at an agent that swallows the error and reports success anyway, and pytest shows you exactly what happened:

```text
AssertionError: answer does not contain 'is confirmed': answer was 'Order 42 is confirmed and paid; receipt sent for order 42.'

Observed
  1. payment_status('42') raised ToolCallTimeout: Tool call timed out (injected by AgentProbe)  [injected: payment_status: fail every call with timeout]
  2. send_receipt('42') returned 'receipt sent for order 42'
  scenarios:
    [exercised] payment_status: fail every call with timeout (required; injected 1/open-ended, 1 call(s) to payment_status)
Asserted
  [fail] answer does not contain 'is confirmed': answer was 'Order 42 is confirmed and paid; receipt sent for order 42.'
Inferred (heuristics over the call log)
  fallbacks: [('payment_status', 'send_receipt')]
```

And an agent that never calls the tool (a cache, a hard-coded answer) fails even if every assertion would pass:

```text
AgentProbe gate FAILED: scenarios 0/1 required exercised, assertions 0/0 required passed
  - scenario 'payment_status: fail every call with timeout' not exercised: the agent never called 'payment_status' (tools called: none). It may have bypassed the tool or used a cached answer.
```

## What you can break

Scenarios are declared per tool with `p.on("tool_name")` and are deterministic: the same calls fail on every run.

| Scenario | Use it to test |
|---|---|
| `fail_first(n)` | Recovery: the first *n* calls fail, then the tool works |
| `fail_times(n).then_succeed()` | The same, spelled out |
| `always_fail()` | Outage: every call fails. Does the agent give up honestly? |
| `returns(value, times=1)` | Wrong, stale or incomplete data instead of an exception |
| `slow(ms=..., times=1)` | Latency, together with `p.run(..., timeout=...)` |
| `with_probability(p, seed=...)` | Random, seeded faults for experiments |

Faults: `"timeout"`, `"rate_limit"`, `"api_error"`, `"malformed_output"`, or any exception class or instance via `with_=`. Rules for one tool apply in order, so `fail_first(1).returns("stale", times=2)` fails call 1, returns stale data for calls 2 and 3, then lets calls through.

## What you can check

| Assertion | Passes when |
|---|---|
| `assert_answer_contains(text)` | The agent produced an answer containing *text* |
| `assert_answer_not_contains(text)` | The agent produced an answer **without** *text*. A crash or `None` never passes |
| `assert_no_crash()` | No exception escaped the agent |
| `assert_within(seconds)` | The run finished in time |
| `assert_called(tool)` / `assert_not_called(tool)` | The tool was or wasn't called; in-flight calls count |
| `assert_scenarios_exercised()` | Every required fault fired (also enforced automatically when the `with` block ends) |

Coming in the next release, and on `main` now: `assert_called_with(tool, *args, **kwargs)`, `assert_call_order([...])`, `assert_called_times(tool, n)` and `assert_called_at_most(tool, n)` check the arguments, order and number of tool calls ([tutorial](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/examples/your_agent/README.md#check-the-tool-calls-themselves-next-release)).

The result keeps what was **observed** (every tool call with arguments, outcome and injected fault), what you **asserted** (pass, fail or unassessed) and what is **inferred** (`result.retries`, `result.fallbacks`) apart. `print(result.summary())` shows all three; `result.to_dict()` is versioned, JSON-safe output for CI artifacts.

## Works with your stack

- **Plain functions**, sync or `async`: pass a list, tuple or `{name: function}` dict to `p.wrap(...)`.
- **LangChain tools** (`@tool`, `StructuredTool`, `BaseTool` subclasses): wrapped as copies with the same schema.
- **LangGraph**: pass the wrapped tools to your graph. The prebuilt ReAct agent re-raises tool exceptions by default; use `ToolNode(tools, handle_tool_errors=True)` if your agent should see the error and recover.
- **Any agent callable** or object with `invoke()`: `p.run(agent, input)`.
- **Async agents and async tests:** `await p.arun(agent, input, timeout=...)`. At the deadline the probe stops injecting and requests cancellation; an agent that ignores it is reported as still running, never as passed.

What AgentProbe guarantees about your tools (never modified, cleanup on exceptions and cancellation, concurrency, agents that outlive their deadline) is in [docs/instrumentation.md](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/docs/instrumentation.md).

### Good to know before you wire in real tools

- **Real tools really run.** A call that gets a fault (`fail_*`, `always_fail`, `returns`) never reaches your tool, but every other call does, including calls delayed by `slow()`. If a tool sends email, charges a card or writes to a database, point it at a sandbox or a test double in your fault tests.
- **Reports are not redacted yet.** Summaries, assertion messages and `to_dict()` include tool arguments, return values and answers as they were. Keep secrets and customer data out of fault-test inputs, or don't upload the JSON as a CI artifact. Automatic redaction is planned ([#102](https://github.com/chhayankjain/agentprobe-toolkit/issues/102)).

## Run it in CI

Fault tests are ordinary pytest tests, so any CI runs them. A ready-made GitHub Actions workflow is in [`examples/your_agent/ci.yml`](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/examples/your_agent/ci.yml):

```yaml
- run: pip install "agentprobe-toolkit>=1.4" pytest
- run: pytest tests/ -m "not integration" --junitxml=agentprobe-results.xml
```

The [10-minute tutorial](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/examples/your_agent/README.md) walks through a complete example agent, shows the same tests catching a deliberately broken version, and sets up CI. The [real-model example](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/examples/real_model_agent/README.md) runs the same kind of tests on a LangChain agent with a local Ollama model, and shows what they caught across models and prompts. The [case study](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/docs/case-studies/agents-from-scratch.md) shows AgentProbe finding an unhandled tool exception in a LangGraph tutorial agent we don't own.

## Try the demo (no model, no key)

With AgentProbe installed as above:

```bash
python -m agentprobe.demo                      # rate-limit faults, p=0.3, 10 runs
python -m agentprobe.demo --fault timeout --p 0.5 --runs 20
python -m agentprobe.demo --json               # machine-readable report
```

The demo runs a small LangGraph agent on a scripted, deterministic model, injects seeded faults into its calculator tool, and prints the metrics, the per-call injection log with OpenTelemetry span ids, and each run's outcome.

Task guides (retries, timeouts and rate limits, wrong data, LangGraph tool errors, CI) and the full reference are on the [docs site](https://chhayankjain.github.io/agentprobe-toolkit/).

## Get involved

- **Have a real agent?** Try AgentProbe on it and tell us where it hurt ([#68](https://github.com/chhayankjain/agentprobe-toolkit/issues/68)). That's the most valuable contribution there is.
- **Want to contribute code?** [Good first issues](https://github.com/chhayankjain/agentprobe-toolkit/labels/good%20first%20issue) are small and need no model or API key. Setup is in [CONTRIBUTING.md](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/CONTRIBUTING.md).
- **Where it's going:** [ROADMAP.md](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/ROADMAP.md). Next up: a pytest plugin, secret redaction in reports and a fault coverage report.
- **Questions and ideas:** [Discussions](https://github.com/chhayankjain/agentprobe-toolkit/discussions).

---

## For researchers: the benchmark and the paper

AgentProbe began as the toolkit behind the paper *"AgentProbe: Systematic Fault Injection and Observability for Evaluating the Reliability of LLM-Based Intelligent Agents"* (submitted to MAKE, MDPI, 2026). The benchmark harness, failure taxonomy, every experiment batch, the paper source and the steps to reproduce its results are in [`research/`](https://github.com/chhayankjain/agentprobe-toolkit/tree/main/research).

## Development

```bash
git clone https://github.com/chhayankjain/agentprobe-toolkit.git && cd agentprobe-toolkit
pip install -e ".[dev,plotting]"                     # no model or API key needed
pytest tests/ examples/your_agent -m "not slow and not integration"
ruff check agentprobe/ tests/ examples/ && ruff format --check agentprobe/ tests/ examples/
```

See [CONTRIBUTING.md](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/CONTRIBUTING.md) for the workflow and [ARCHITECTURE.md](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/ARCHITECTURE.md) for how the code fits together.

## Citation

```bibtex
@misc{jain2026agentprobe,
  title={AgentProbe: Systematic Fault Injection and Observability for Evaluating the Reliability of LLM-Based Intelligent Agents},
  author={Jain, Chhayank and Pattnaik, Sarthak and Pinsky, Eugene},
  year={2026},
  note={Manuscript under review}
}
```

## License

MIT — see [LICENSE](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/LICENSE).
