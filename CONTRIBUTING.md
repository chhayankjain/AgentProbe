# Contributing to AgentProbe

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone and install
git clone https://github.com/chhayankjain/agentprobe-toolkit.git
cd agentprobe-toolkit
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,plotting]"   # no model or API key needed for the offline suite

# Run tests
pytest tests/ examples/your_agent -v -m "not slow and not integration"

# Run linter
ruff check agentprobe/ tests/ examples/
ruff format --check agentprobe/ tests/ examples/

# Type check (strict; CI fails on any error)
mypy agentprobe/
```

## Where to start

- [ROADMAP.md](ROADMAP.md) explains what we're building and why, milestone by milestone.
- [Good first issues](https://github.com/chhayankjain/agentprobe-toolkit/labels/good%20first%20issue) are small and need no model or API key.
- Every roadmap issue has **Why / Scope / Acceptance criteria / Where to start / Size**. Comment to claim one. We aim to reply within two working days; see the [maintainer service targets](ROADMAP.md#maintainer-service-targets), including what happens when maintainers are away.
- Have a real agent? Trying AgentProbe on it and reporting friction (#68) is the most valuable contribution right now.

## Workflow

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add or update tests as needed
4. Run the offline tests and lint/format checks above
5. Open a pull request against `main`

## Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Line length: 100 characters
- Type hints are encouraged but not strictly required
- Keep changes focused — one feature or fix per PR

## Running Experiments

Experiments require [Ollama](https://ollama.ai) running locally:

```bash
pip install -e ".[ollama]"
ollama pull llama3.1
python research/experiments/run_experiment_v2.py --frameworks langgraph --model llama3.1 --seed 42
```

## Docs site

The docs site (https://agentprobe-toolkit.pages.dev/) is built with MkDocs from `docs/` and `mkdocs.yml`, and deployed to Cloudflare Pages from `main` by `.github/workflows/docs.yml` (repository secrets `CLOUDFLARE_API_TOKEN`, with only *Cloudflare Pages: Edit*, and `CLOUDFLARE_ACCOUNT_ID`). The old GitHub Pages address serves redirects to it. Running the workflow by hand on a branch deploys a preview for that branch. To preview it locally:

```bash
pip install -e ".[docs]"
mkdocs serve        # http://127.0.0.1:8000
mkdocs build --strict
```

Code blocks fenced as ```` ```python ```` in `docs/getting-started.md` and `docs/guides/` are **run by the test suite** (`tests/test_docs_examples.py`): each page's blocks execute in order and every `test_*` function they define is called. Keep them self-contained. Fence illustrative fragments (code that imports your own agent, for example) as ```` ```py ````; they render the same and are not run.

## Reporting Issues

- Use the [bug report template](https://github.com/chhayankjain/agentprobe-toolkit/issues/new?template=bug_report.yml) for bugs
- Use the [feature request template](https://github.com/chhayankjain/agentprobe-toolkit/issues/new?template=feature_request.yml) for ideas

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
