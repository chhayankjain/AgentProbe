<p align="center">
  <img src="https://raw.githubusercontent.com/chhayankjain/agentprobe-toolkit/main/docs/assets/logo.svg" alt="" width="72">
</p>

<h1 align="center">AgentProbe</h1>

<p align="center">
  <strong>Fault-injection testing for tool-using LLM agents, in pytest.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/agentprobe-toolkit/"><img src="https://img.shields.io/pypi/v/agentprobe-toolkit.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/agentprobe-toolkit/"><img src="https://img.shields.io/pypi/pyversions/agentprobe-toolkit.svg" alt="Python versions"></a>
  <a href="https://github.com/chhayankjain/agentprobe-toolkit/actions/workflows/ci.yml"><img src="https://github.com/chhayankjain/agentprobe-toolkit/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://agentprobe-toolkit.pages.dev/"><img src="https://img.shields.io/badge/docs-agentprobe--toolkit.pages.dev-4f46e5.svg" alt="Documentation"></a>
  <a href="https://github.com/chhayankjain/agentprobe-toolkit/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT license"></a>
</p>

<p align="center">
  <a href="https://agentprobe-toolkit.pages.dev/"><strong>Documentation</strong></a> ·
  <a href="https://agentprobe-toolkit.pages.dev/getting-started/">Getting started</a> ·
  <a href="https://agentprobe-toolkit.pages.dev/guides/retries/">Guides</a> ·
  <a href="https://agentprobe-toolkit.pages.dev/reference/scenarios/">Reference</a> ·
  <a href="https://github.com/chhayankjain/agentprobe-toolkit/blob/main/CHANGELOG.md">Changelog</a>
</p>

---

Tools fail in production: they time out, hit rate limits and return bad data. AgentProbe wraps the tools your agent already uses, makes them fail on purpose and deterministically, and checks what the agent did about it.

- **Deterministic scenarios.** Fail specified tool calls predictably, without relying on random outages. Live-model responses and external tools can still vary.
- **Evidence-based.** A test fails unless every fault you declared actually fired. An agent that skips the tool or answers from a cache cannot pass by accident.
- **Non-invasive.** Your agent and tools run unchanged; AgentProbe steps in only at the tool boundary, on copies of your tools.
- **Just pytest.** No model, API key or service is needed for scripted agents, so fault tests run in any CI.

## Installation

```bash
pip install "agentprobe-toolkit>=1.4" pytest
```

The package installs as `agentprobe-toolkit` and imports as `agentprobe`. Requires Python 3.11+.

## Quick start

Given an agent `answer(question, tools)` and a dict `TOOLS` of the functions it calls, this test checks that the agent admits a payment-service outage instead of claiming success:

```python
from agentprobe.testing import probe

from my_agent import TOOLS, answer


def test_admits_failure_when_payment_service_is_down():
    with probe() as p:
        tools = p.wrap(TOOLS)  # probed copies; your tools are untouched
        p.on("payment_status").always_fail(with_="timeout")

        result = p.run(lambda q: answer(q, tools), "Did order 42 go through?")

        result.assert_no_crash()
        result.assert_answer_not_contains("is confirmed")
        result.assert_answer_contains("couldn't confirm")
        result.assert_not_called("send_receipt")
```

Run it with `pytest`. If the agent swallows the error and reports success, the failure shows what happened, what was asserted and what was inferred:

```text
AssertionError: answer does not contain 'is confirmed': answer was 'Order 42 is confirmed and paid; receipt sent for order 42.'

Observed
  final answer: 'Order 42 is confirmed and paid; receipt sent for order 42.'
  elapsed: 0.00s
  1. payment_status('42') raised ToolCallTimeout: Tool call timed out (injected by AgentProbe)  [injected: payment_status: fail every call with timeout]
  2. send_receipt('42') returned 'receipt sent for order 42'
  scenarios:
    [exercised] payment_status: fail every call with timeout (required; injected 1/open-ended, 1 call(s) to payment_status)
Asserted
  [pass] no unhandled exception: none raised
  [fail] answer does not contain 'is confirmed': answer was 'Order 42 is confirmed and paid; receipt sent for order 42.'
Inferred (heuristics over the call log)
  retries: none
  fallbacks: [('payment_status', 'send_receipt')]
```

If the agent never calls `payment_status` at all (a cache, a hard-coded answer), the test still fails: AgentProbe's gate reports the scenario as *not exercised*, even when every assertion passes.

The example imports your own agent. For a complete runnable example with no model or API key, follow [Getting started](https://agentprobe-toolkit.pages.dev/getting-started/).

## Features

**Fault scenarios**, declared per tool and applied in order:

| Scenario | Simulates |
|---|---|
| `fail_first(n)`, `fail_times(n).then_succeed()` | Transient failures, to test retries and recovery |
| `always_fail()` | An outage, to test that the agent gives up honestly |
| `returns(value)` | Wrong, stale or incomplete data |
| `slow(ms=...)` | Latency, combined with a deadline: `p.run(..., timeout=...)` |
| `with_probability(p, seed=...)` | Random, reproducible faults for experiments |

Built-in faults are `"timeout"`, `"rate_limit"`, `"api_error"` and `"malformed_output"`; pass any exception class or instance with `with_=` to match what your code really handles.

**Assertions** on the answer (`assert_answer_contains`, `assert_answer_not_contains`), on the run (`assert_no_crash`, `assert_within`) and on tool calls (`assert_called`, `assert_not_called`). On `main` for the next release: regex and predicate answer checks (`assert_answer_matches`, `assert_answer(check)`) that tolerate varied wording, and argument, order and call-count assertions. Every result separates what was **observed**, what was **asserted** and what was **inferred**, and exports versioned JSON with `result.to_dict()`.

See the [scenario](https://agentprobe-toolkit.pages.dev/reference/scenarios/) and [assertion](https://agentprobe-toolkit.pages.dev/reference/assertions/) reference for details.

## Integrations

| Works with | How |
|---|---|
| Plain Python functions, sync or `async` | `p.wrap(TOOLS)` with a dict, list or single function |
| LangChain tools (`@tool`, `StructuredTool`, `BaseTool`) | Wrapped as copies with the same name and schema |
| LangGraph | Pass the wrapped tools to your graph ([guide](https://agentprobe-toolkit.pages.dev/guides/langgraph-tool-errors/)) |
| Any agent callable or object with `invoke()` | `p.run(agent, input)`, or `await p.arun(...)` for async agents |

How wrapping, concurrency, deadlines and cleanup work is documented in [How probing works](https://agentprobe-toolkit.pages.dev/instrumentation/).

## Learn more

- [**Guides**](https://agentprobe-toolkit.pages.dev/guides/retries/): retries, timeouts and rate limits, wrong data, LangGraph tool errors, real exception types, CI.
- [**Tutorial**](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/examples/your_agent/README.md): add a fault test to an existing agent and run it in CI, in ten minutes.
- [**Real-model example**](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/examples/real_model_agent/README.md): the same tests on a LangChain agent with a local Ollama model.
- [**Case study**](https://agentprobe-toolkit.pages.dev/case-studies/agents-from-scratch/): fault tests that found an unhandled tool exception in a third-party LangGraph agent.
- **Demo:** `python -m agentprobe.demo` runs a scripted LangGraph agent under seeded faults, with no model or API key.

## Before you test real tools

- **Real tools run.** Exception and substituted-return faults bypass the original tool. Other calls, including calls delayed by `slow()`, execute it. Use sandboxes or test doubles for payments, email and other side effects.
- **Reports are not redacted yet.** Tool arguments, results and answers appear verbatim in reports and JSON; keep secrets out of test inputs ([#102](https://github.com/chhayankjain/agentprobe-toolkit/issues/102)).

## Contributing

Contributions are welcome. [Good first issues](https://github.com/chhayankjain/agentprobe-toolkit/labels/good%20first%20issue) need no model or API key; setup and workflow are in [CONTRIBUTING.md](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/CONTRIBUTING.md), and the plan is in [ROADMAP.md](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/ROADMAP.md). If you use AgentProbe on a real agent, feedback in [#68](https://github.com/chhayankjain/agentprobe-toolkit/issues/68) or [Discussions](https://github.com/chhayankjain/agentprobe-toolkit/discussions) is especially valuable.

## Research

AgentProbe began as the toolkit behind the paper *AgentProbe: Systematic Fault Injection and Observability for Evaluating the Reliability of LLM-Based Intelligent Agents* (under review). The benchmark, failure taxonomy, experiment data and reproduction steps are in [`research/`](https://github.com/chhayankjain/agentprobe-toolkit/tree/main/research).

```bibtex
@misc{jain2026agentprobe,
  title={AgentProbe: Systematic Fault Injection and Observability for Evaluating the Reliability of LLM-Based Intelligent Agents},
  author={Jain, Chhayank and Pattnaik, Sarthak and Pinsky, Eugene},
  year={2026},
  note={Manuscript under review}
}
```

## License

[MIT](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/LICENSE)
