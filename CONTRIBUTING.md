# Contributing to AgentProbe

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone and install
git clone https://github.com/chhayankjain/AgentProbe.git
cd AgentProbe
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,plotting,ollama]"   # installs the agentprobe-toolkit distribution in editable mode

# Run tests
pytest tests/ -v

# Run linter
ruff check agentprobe/ tests/
ruff format --check agentprobe/ tests/
```

## Workflow

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add or update tests as needed
4. Ensure `pytest tests/ -v` passes and `ruff check` is clean
5. Open a pull request against `main`

## Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Line length: 100 characters
- Type hints are encouraged but not strictly required
- Keep changes focused — one feature or fix per PR

## Running Experiments

Experiments require [Ollama](https://ollama.ai) running locally:

```bash
ollama pull llama3.1
python experiments/run_experiment_v2.py --frameworks langgraph --model llama3.1 --seed 42
```

## Reporting Issues

- Use the [bug report template](https://github.com/chhayankjain/AgentProbe/issues/new?template=bug_report.yml) for bugs
- Use the [feature request template](https://github.com/chhayankjain/AgentProbe/issues/new?template=feature_request.yml) for ideas

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
