---
description: Run LLM agent fault tests in GitHub Actions or any CI. AgentProbe tests are ordinary pytest tests and need no model or API key with scripted agents.
---

# Run fault tests in CI

Fault tests are ordinary pytest tests, so any CI runs them. With plain-Python or scripted agents they need no model and no API key.

## GitHub Actions

Copy this to `.github/workflows/agent-fault-tests.yml`:

```yaml
name: Agent fault tests

on:
  push:
  pull_request:

jobs:
  fault-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install "agentprobe-toolkit>=1.4" pytest
      # Point this at the folder with your fault tests.
      - run: pytest tests/ --junitxml=agentprobe-results.xml
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: agentprobe-results
          path: agentprobe-results.xml
```

The same file is in the repository as [`examples/your_agent/ci.yml`](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/examples/your_agent/ci.yml).

## Real models in CI

Tests that call a real model are slower, cost money and can vary between runs. A common split:

- **Every push:** scripted or plain-Python agents, deterministic faults. Fast and free.
- **Nightly or on demand:** the same tests against a real model, marked with a pytest marker (for example `@pytest.mark.integration`) and run with `pytest -m integration`.

The [real-model example](https://github.com/chhayankjain/agentprobe-toolkit/blob/main/examples/real_model_agent/README.md) shows this split with a LangChain agent on a local Ollama model, and checks recovery on the call log rather than on wording.

## Keeping evidence

`result.to_dict()` returns versioned, JSON-safe output: the call log, scenario evidence, assertion results and the gate verdict. Write it to a file and upload it as an artifact to keep evidence of what each test exercised.

!!! warning
    Reports are not redacted yet: tool arguments, return values and answers appear verbatim. Keep secrets and customer data out of fault-test inputs, or don't upload the JSON ([#102](https://github.com/chhayankjain/agentprobe-toolkit/issues/102)).
