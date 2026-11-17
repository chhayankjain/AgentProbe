# Roadmap

**Goal:** developers keep AgentProbe tests in their CI because they catch failures they care about.

**Who it's for first:** Python developers building tool-using agents who already use pytest. Their question is concrete: *will my agent still complete the task when this tool fails?*

Everything below serves that one question. Research use (the paper's experiments) stays fully supported, but it is a separate track and does not set priorities.

Live tracking: [milestones](https://github.com/chhayankjain/AgentProbe/milestones) and the pinned [roadmap issue #38](https://github.com/chhayankjain/AgentProbe/issues/38).

---

## Principles

1. **Test the agent you already have.** No replacement agent architecture, special backend, or observability server required.
2. **Prove the fault happened.** A fault test that never reached its tool is reported as *not exercised* and fails, never silently passes. An agent that bypasses a tool must not look resilient.
3. **Deterministic by default.** Regression tests get the same failure every run. Probability-based injection remains for experiments.
4. **Separate what we saw from what we infer.** Results keep observed events (tool calls, injected faults, exceptions, elapsed time), your assertions (pass, fail or unassessed) and inferred labels (retry, fallback) apart. A deadline being exceeded is reported separately from execution actually stopping.
5. **Leave no trace.** Instrumentation is restored after every test, including after exceptions and cancellation, so one test can't change another.
6. **Gate on requirements, not scores.** CI fails on explicit required assertions and scenario coverage. A weighted score can hide a correctness regression.
7. **Honest support levels.** Every integration is labelled experimental, offline-tested, live-qualified or legacy, and a level is only raised with evidence.
8. **Stable and easy to contribute to.** Semantic versioning with a time-bound deprecation window, fast tests with no model, and small well-scoped issues.

## How we'll know it's working

| Signal | Target |
|---|---|
| Time from `pip install` to a passing fault test on your own agent | under 10 minutes, measured with design partners |
| Design partners who add a fault test to their own agent | 5 |
| **Retention:** design partners still running meaningful fault tests in CI four weeks later | at least 3 of 5 |
| Repositories outside the maintainers with AgentProbe tests in CI | 25 within six months of v1.5 |
| Issues and PRs opened by people outside the maintainers | at least 5 per month by the end of v1.5 |

---

## Order of work

Scenario and evidence contract → minimal pytest workflow → design-partner usage → fix observed friction → richer reporting and silent-failure scenarios → one additional integration.

External usage decides what comes after v1.4.

## Milestones

### v1.4 — Test your own agent

The smallest end-to-end workflow.

*Exit criterion: an external developer adds a fault test to their existing agent, shows that it catches a deliberately broken behaviour, and runs it in CI.*

| | Issue |
|---|---|
| Scenario evidence: prove every required fault actually fired; *not exercised* fails | #94 |
| Instrumentation contract: state, reset, concurrency, guaranteed cleanup | #95 |
| Wrap the tools of an existing agent: plain callables and LangChain tools | #63 |
| Deterministic scenarios: fail first N calls, fail then succeed, slow, return a value | #65 |
| `ProbeResult`: observed events, assertion results and inferred labels kept separate | #66 |
| Minimal pytest workflow: a helper, one outcome assertion, one CI recipe | #96 |
| Tutorial: add a fault test to your existing agent and run it in CI | #67 |
| Design partners: five developers, retention checked at four weeks | #68 |

### v1.5 — Friction fixes, pytest plugin and supported integrations

*Exit criterion: the friction design partners reported is fixed, fault tests use a first-class pytest plugin, and every advertised integration has an honest support level.*

| | Issue |
|---|---|
| Fixes for friction found by design partners | filed from #68 |
| pytest plugin: `probe` fixture and marker, shipped in the package | #69 |
| Outcome assertions: no crash, answer contains, within deadline, retry budget, admits failure | #70 |
| Scripted mode for CI, live mode for real models, cost guards, pass-rate thresholds | #71 |
| Failure reports: terminal diagnostics and versioned JSON (pytest's JUnit for CI viewers) | #72 |
| `agentprobe check`: gate on required assertions and scenario coverage | #74 |
| Graph support for documented LangGraph shapes with explicit tool selection | #64 |
| Adapter contract and conformance test suite | #78 |
| Integration support levels: experimental, offline-tested, live-qualified, legacy | #97 |
| Live qualification before raising a support level | #92 |
| API stability and deprecation policy; `py.typed` | #86 |
| More CI recipes; a packaged GitHub Action only on demand | #73 |

### v1.6 — Silent failures

*Exit criterion: AgentProbe can tell you when your agent confidently used bad tool data.*

| | Issue |
|---|---|
| Stale and wrong-value results | #75 |
| Schema drift: missing, renamed or retyped fields | #76 |
| Corrupted-but-valid tool output | #28 |
| Schedules beyond per-call probability: bursts, intermittent, persistent | #27 |
| Did the agent notice? Graders and `assert_noticed_bad_data()` | #77 |

### v1.7 — One more integration

Exactly one, chosen by design-partner demand. Candidates (label `candidate integration`); the others stay open for contributors:

| Candidate | Issue |
|---|---|
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
| Governance, CODEOWNERS, contributor ladder, recognition | #87 |
| Discussions categories and welcome post | #88 |
| Strict type checking | #29 |
| Coverage tracking | #30 |
| Integration tests for the full pipeline | #36 |

### Research track (not on the adoption critical path)

| | Issue |
|---|---|
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

Browse all [good first issues](https://github.com/chhayankjain/AgentProbe/labels/good%20first%20issue). Comment on an issue to claim it. Have a real agent? #68 is the most valuable contribution of all: try it and tell us where it hurt.

## What we're deliberately not doing yet

- **An aggregate resilience score.** CI gates on explicit requirements first (#74). An optional score can follow once users have comparable profiles.
- **A packaged GitHub Action.** Recipes first (#96, #73).
- **Markdown and HTML reports.** Terminal and versioned JSON first (#72).
- **More than one new integration at a time.** Design partners choose (#79, #80, #32, #33).
- **A public leaderboard.** Expensive, easy to game, and it conflates adapter quality with model quality. Versioned reference results come first.
- **A general plugin registry.** Designed after several real integrations show what the interface needs (#78 comes first).
- **Security and prompt-injection fuzzing.** A different problem with different users. AgentProbe stays focused on reliability.
- **Dashboards** (#31). Reports in CI come first.

These can change when users ask for them. Open a [Discussion](https://github.com/chhayankjain/AgentProbe/discussions) with your use case.

## Maintainer service targets

These are targets, not guarantees, and we publish when we miss them.

- **First reply** on issues and PRs within two working days.
- **Releases** every two to four weeks while a milestone is active, each with a changelog entry for every user-visible change.
- **Deprecations:** a deprecated public API keeps working, with a warning, for at least one minor release **and** at least 60 days, whichever is longer.
- **Absence policy:** if maintainers are unavailable for more than a week, a note goes in the pinned roadmap issue (#38) with the expected return date, and review targets pause. Security reports follow `SECURITY.md` regardless.
- **Credit:** every merged contribution is credited in the release notes.
