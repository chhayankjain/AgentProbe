# Roadmap

**Goal:** developers keep AgentProbe tests in their CI because they catch failures they care about.

**Who it's for first:** Python developers building tool-using agents who already use pytest. Their question is concrete: *will my agent still complete the task when this tool fails?*

Everything below serves that one question. Research use (the paper's experiments) stays fully supported, but it is a separate track and does not set priorities.

Live tracking: [milestones](https://github.com/chhayankjain/AgentProbe/milestones) and the pinned [roadmap issue #38](https://github.com/chhayankjain/AgentProbe/issues/38).

---

## Principles

1. **Test the agent you already have.** No replacement agent architecture, special backend, or observability server required.
2. **Deterministic by default.** Regression tests get the same failure every run. Probability-based injection remains for experiments.
3. **Explain what happened.** Every run answers: which tool failed, did the agent retry or fall back, was the final answer correct, did it stay within its deadline and retry budget.
4. **Outcomes over internals.** Assertions check what users care about, not framework details.
5. **Stable and boring to upgrade.** Semantic versioning, a deprecation window, and no surprise breaking changes.
6. **Easy to contribute to.** Fast tests with no model, small well-scoped issues, and replies within two working days.

## How we'll know it's working

| Signal | Target |
|---|---|
| Time from `pip install` to a passing fault test on your own agent | under 10 minutes |
| Developers outside the maintainers with AgentProbe tests in their repo | 5, then 25 |
| Those developers coming back to add another scenario | most of them |
| Issues and PRs from people we don't know | steady month over month |
| First reply on issues and PRs | within 2 working days |
| Release cadence | every 2–4 weeks |

---

## Milestones

### v1.4 — Test your own agent

*Exit criterion: a developer goes from `pip install` to a passing fault test on their own agent in under 10 minutes.*

| | Issue |
|---|---|
| Wrap the tools of an existing agent: plain callables and LangChain tools | #63 |
| Probe an existing compiled LangGraph graph as-is | #64 |
| Deterministic scenarios: fail first N calls, fail then succeed, slow, return a value | #65 |
| `ProbeResult`: which tool failed, retries, fallback, final answer, deadline | #66 |
| Tutorial: test your own agent in 10 minutes | #67 |
| Design partners: five developers try it on real agents; we fix the friction | #68 |

### v1.5 — pytest, CI and failure reports

*Exit criterion: fault tests run in pytest and CI like any other test, and a failing test explains itself.*

| | Issue |
|---|---|
| pytest plugin: `probe` fixture and marker, shipped in the package | #69 |
| Outcome assertions: no crash, answer contains, within deadline, retry budget, admits failure | #70 |
| Scripted mode for CI, live mode for real models, cost guards, pass-rate thresholds | #71 |
| Readable failure reports: narrative, call log, Markdown, JSON, JUnit | #72 |
| GitHub Action and CI recipes | #73 |
| Resilience score and `agentprobe check` with a CI exit code | #74 |

### v1.6 — Silent failures

*Exit criterion: AgentProbe can tell you when your agent confidently used bad tool data.*

| | Issue |
|---|---|
| Stale and wrong-value results | #75 |
| Schema drift: missing, renamed, or retyped fields | #76 |
| Corrupted-but-valid tool output | #28 |
| Schedules beyond per-call probability: bursts, intermittent, persistent | #27 |
| Did the agent notice? Graders and `assert_noticed_bad_data()` | #77 |

### v1.7 — Ecosystem

*Exit criterion: adding a framework is a well-defined, reviewable contribution.*

| | Issue |
|---|---|
| Adapter contract and conformance test suite | #78 |
| Modern AutoGen adapter; pyautogen 0.1.14 moves to `agentprobe.legacy` (kept for paper reproduction) | #79 |
| MCP fault proxy: test any agent that uses tools over MCP | #80 |
| OpenAI Agents SDK adapter | #33 |
| CrewAI adapter | #32 |

### Contributor experience (ongoing, alongside every milestone)

| | Issue |
|---|---|
| CONTRIBUTING rewritten around first contributions | #81 |
| Dev container and Codespaces, no Ollama needed | #82 |
| One-command dev tasks (`nox`) matching CI | #83 |
| Issue templates: new fault scenario, new adapter, first-test feedback | #84 |
| Documentation site with tutorials and API reference | #85 |
| API stability and deprecation policy; `py.typed` | #86 |
| Governance, CODEOWNERS, contributor ladder, recognition | #87 |
| Discussions categories and welcome post | #88 |
| Strict type checking | #29 |
| Coverage tracking | #30 |
| Integration tests for the full pipeline | #36 |

### Research track (not on the adoption critical path)

| | Issue |
|---|---|
| Live qualification on hosted models before claiming support | #92 |
| Orchestration fault experiments | #24 |
| Context degradation experiments | #25 |
| Multi-seed experiments | #34 |
| JSON Schema for experiment outputs | #35 |

---

## Start contributing

New here? These are small, self-contained, and need no model or API key:

- #89 — `--json` output for the demo
- #90 — Glossary of metrics and statuses
- #91 — Example: a retry policy under timeout vs rate-limit faults
- #82 — Dev container and Codespaces
- #83 — `nox` dev tasks
- #84 — New issue templates

Browse all [good first issues](https://github.com/chhayankjain/AgentProbe/labels/good%20first%20issue). Comment on an issue to claim it; we'll reply within two working days. Have a real agent? #68 is the most valuable contribution of all: try it and tell us where it hurt.

## What we're deliberately not doing yet

- **A public leaderboard.** Expensive, easy to game, and it conflates adapter quality with model quality. Versioned reference results come first.
- **A general plugin registry.** We'll design one after several real integrations show what the interface needs (#78 comes first).
- **Many new frameworks at once.** Each one must pass the conformance suite.
- **Security and prompt-injection fuzzing.** A different problem with different users. AgentProbe stays focused on reliability.
- **Dashboards** (#31). Reports in CI come first.

These can change when users ask for them. Open a [Discussion](https://github.com/chhayankjain/AgentProbe/discussions) with your use case.

## Maintainer commitments

- First reply on every issue and PR within two working days.
- A release every two to four weeks, with a changelog entry for every user-visible change.
- No breaking change without a deprecation warning for at least one minor release.
- Every merged contribution is credited in the release notes.
