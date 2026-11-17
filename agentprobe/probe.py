"""Test the agent you already have: deterministic tool faults with proof they fired.

Wrap your agent's tools, declare which faults to apply, run the agent, and
check the outcome::

    from agentprobe.testing import probe

    def test_checkout_survives_a_timeout():
        with probe() as p:
            tools = p.wrap([payment_status, send_receipt])
            p.on("payment_status").fail_first(1, with_="timeout")
            result = p.run(lambda q: my_agent(q, tools), "Did order 42 go through?")
            result.assert_answer_contains("confirmed")
        # Leaving the block enforces the gate: every required scenario must have
        # been exercised and every required assertion must have passed.

Design contract (see ``docs/instrumentation.md``):

* **Your tools are never modified.** :meth:`Probe.wrap` returns new objects (a new
  function, or a shallow copy of a LangChain tool). Originals stay byte-for-byte
  as they were, so two probes on the same tool cannot interfere.
* **Evidence, not hope.** Each scenario records whether its tool was reached and
  whether its fault was injected. A required scenario that never fired is
  ``not_exercised`` and fails the gate.
* **Observed, asserted and inferred are separate.** :class:`ProbeResult` keeps the
  call log, your assertion results (``pass``/``fail``/``unassessed``/``error``)
  and heuristic labels (retry, fallback) apart.
* **Still-running agents are reported, not hidden.** With ``timeout=`` the agent
  runs in a worker thread that Python cannot kill. If it outlives the deadline the
  result says so, the probe closes, and any later tool call passes through
  without a fault and is recorded as ``after_probe_end``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import copy
import functools
import inspect
import random
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from agentprobe.agents.base import message_text
from agentprobe.injector.tool_failure import (
    ToolCallAPIError,
    ToolCallError,
    ToolCallMalformedOutput,
    ToolCallRateLimited,
    ToolCallTimeout,
)
from agentprobe.observer.tracer import get_tracer

SCHEMA_VERSION = "1.0"

_tracer = get_tracer(__name__)

#: Guards against counting one logical tool call twice when a wrapped async
#: method delegates to a wrapped sync method on the same tool.
_IN_WRAPPER: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "agentprobe_in_wrapper", default=False
)


class ProbeError(Exception):
    """Misuse of the probe API (bad tool, bad scenario, wrong lifecycle)."""


class ProbeGateError(AssertionError):
    """A required scenario was not exercised or a required assertion did not pass."""


# ---------------------------------------------------------------------------
# Faults
# ---------------------------------------------------------------------------

_FAULTS: dict[str, Callable[[], ToolCallError]] = {
    "timeout": lambda: ToolCallTimeout("Tool call timed out (injected by AgentProbe)"),
    "rate_limit": lambda: ToolCallRateLimited(retry_after=1.0),
    "api_error": lambda: ToolCallAPIError(status_code=503),
    "malformed_output": lambda: ToolCallMalformedOutput(
        "Tool returned unparseable output (injected by AgentProbe)"
    ),
}


def _fault_factory(with_: str | type[BaseException] | BaseException) -> Callable[[], BaseException]:
    if isinstance(with_, str):
        try:
            return _FAULTS[with_]
        except KeyError:
            raise ProbeError(
                f"Unknown fault {with_!r}; use one of {sorted(_FAULTS)} or an exception"
            ) from None
    if isinstance(with_, BaseException):
        return lambda: copy.copy(with_)
    if isinstance(with_, type) and issubclass(with_, BaseException):
        return lambda: with_(f"{with_.__name__} injected by AgentProbe")
    raise ProbeError(f"with_ must be a fault name, an exception class or instance, not {with_!r}")


def _fault_label(with_: Any) -> str:
    if isinstance(with_, str):
        return with_
    if isinstance(with_, BaseException):
        return type(with_).__name__
    return getattr(with_, "__name__", repr(with_))


# ---------------------------------------------------------------------------
# Scenario rules
# ---------------------------------------------------------------------------


@dataclass
class _Action:
    kind: str  # "raise" | "delay" | "return"
    exc: BaseException | None = None
    seconds: float = 0.0
    value: Any = None


@dataclass
class _Rule:
    rule_id: int
    tool: str
    description: str
    required: bool
    planned: int | None  # faults the rule intends to apply; None = open-ended
    decide: Callable[[int], _Action | None]  # nth call offered to this rule -> action
    injected: int = 0
    offered: int = 0  # calls that reached this rule (not consumed by earlier rules)


class ToolScenarios:
    """Builder returned by :meth:`Probe.on`. Each method adds one scenario rule.

    All rules are deterministic except :meth:`with_probability`. Rules for the
    same tool form a sequence: each call is offered to the rules in declaration
    order, and each rule counts only the calls that reach it. So
    ``fail_first(1).returns("stale", times=2)`` fails call 1, returns stale data
    for calls 2 and 3, and lets call 4 through.
    """

    def __init__(self, probe: Probe, tool: str) -> None:
        self._probe = probe
        self._tool = tool

    def fail_first(
        self, n: int = 1, *, with_: Any = "timeout", required: bool = True
    ) -> ToolScenarios:
        """Fail the first *n* calls to this tool, then let calls through."""
        if n < 1:
            raise ProbeError("fail_first(n) needs n >= 1")
        make = _fault_factory(with_)
        self._probe._add_rule(
            self._tool,
            f"{self._tool}: fail {_first_calls(n)} with {_fault_label(with_)}",
            required,
            n,
            lambda i: _Action("raise", exc=make()) if i <= n else None,
        )
        return self

    def fail_times(self, n: int, *, with_: Any = "timeout", required: bool = True) -> _Then:
        """``fail_times(2).then_succeed()``: an explicit spelling of :meth:`fail_first`."""
        self.fail_first(n, with_=with_, required=required)
        return _Then(self)

    def always_fail(self, *, with_: Any = "timeout", required: bool = True) -> ToolScenarios:
        """Fail every call to this tool."""
        make = _fault_factory(with_)
        self._probe._add_rule(
            self._tool,
            f"{self._tool}: fail every call with {_fault_label(with_)}",
            required,
            None,
            lambda i: _Action("raise", exc=make()),
        )
        return self

    def slow(self, *, ms: float, times: int | None = 1, required: bool = True) -> ToolScenarios:
        """Delay the first *times* calls by *ms* milliseconds, then run them normally.
        ``times=None`` delays every call."""
        if ms < 0:
            raise ProbeError("slow(ms) needs ms >= 0")
        seconds = ms / 1000.0
        label = "every call" if times is None else _first_calls(times)
        self._probe._add_rule(
            self._tool,
            f"{self._tool}: delay {label} by {ms:g} ms",
            required,
            times,
            lambda i: _Action("delay", seconds=seconds) if times is None or i <= times else None,
        )
        return self

    def returns(self, value: Any, *, times: int | None = 1, required: bool = True) -> ToolScenarios:
        """Return *value* instead of calling the tool for the first *times* calls
        (``None`` = every call). Use it for wrong, stale or incomplete data."""
        label = "every call" if times is None else _first_calls(times)
        self._probe._add_rule(
            self._tool,
            f"{self._tool}: return {value!r} for {label}",
            required,
            times,
            lambda i: (
                _Action("return", value=copy.deepcopy(value))
                if times is None or i <= times
                else None
            ),
        )
        return self

    def with_probability(
        self, p: float, *, with_: Any = "timeout", seed: int | None = None, required: bool = True
    ) -> ToolScenarios:
        """Fail each call with probability *p* (seeded). For experiments; regression
        tests should prefer the deterministic rules. Exercised once at least one
        fault fired."""
        if not 0.0 <= p <= 1.0:
            raise ProbeError("probability must be in [0, 1]")
        rng = random.Random(seed)
        make = _fault_factory(with_)
        self._probe._add_rule(
            self._tool,
            f"{self._tool}: fail with {_fault_label(with_)} at p={p:g}"
            + (f" (seed {seed})" if seed is not None else ""),
            required,
            None,
            lambda i: _Action("raise", exc=make()) if rng.random() < p else None,
        )
        return self


def _first_calls(n: int) -> str:
    return f"the first {n} call{'s' if n > 1 else ''}"


class _Then:
    def __init__(self, scenarios: ToolScenarios) -> None:
        self._scenarios = scenarios

    def then_succeed(self) -> ToolScenarios:
        return self._scenarios


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@dataclass
class CallRecord:
    """One tool call seen through a wrapper. Observed, never inferred."""

    index: int  # order across all tools in this probe (1-based)
    tool: str
    tool_call_index: int  # nth call to this tool (1-based); 0 for after_probe_end
    args: list[Any]
    kwargs: dict[str, Any]
    outcome: str  # "returned" | "raised"
    value: str | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    injected: bool = False
    fault: str | None = None  # scenario description when injected
    rule_id: int | None = None
    duration_ms: float = 0.0
    span_id: str | None = None
    after_probe_end: bool = False


@dataclass
class ScenarioEvidence:
    """Whether one scenario rule actually fired."""

    rule_id: int
    tool: str
    description: str
    required: bool
    planned: int | None
    injected: int
    tool_calls: int
    boundary_reached: bool
    status: str  # "exercised" | "partially_exercised" | "not_exercised"


@dataclass
class AssertionRecord:
    name: str
    status: str  # "pass" | "fail" | "unassessed" | "error"
    message: str
    required: bool = True


@dataclass
class GateReport:
    passed: bool
    failures: list[str]
    scenarios_exercised: int
    scenarios_required: int
    assertions_passed: int
    assertions_required: int

    def __str__(self) -> str:
        head = "PASSED" if self.passed else "FAILED"
        lines = [
            f"AgentProbe gate {head}: scenarios "
            f"{self.scenarios_exercised}/{self.scenarios_required} required exercised, "
            f"assertions {self.assertions_passed}/{self.assertions_required} required passed"
        ]
        lines += [f"  - {f}" for f in self.failures]
        return "\n".join(lines)


def _evidence(rule: _Rule, calls_to_tool: int) -> ScenarioEvidence:
    if rule.planned is None:
        status = "exercised" if rule.injected >= 1 else "not_exercised"
    elif rule.injected >= rule.planned:
        status = "exercised"
    elif rule.injected > 0:
        status = "partially_exercised"
    else:
        status = "not_exercised"
    return ScenarioEvidence(
        rule_id=rule.rule_id,
        tool=rule.tool,
        description=rule.description,
        required=rule.required,
        planned=rule.planned,
        injected=rule.injected,
        tool_calls=calls_to_tool,
        boundary_reached=calls_to_tool > 0,
        status=status,
    )


def _short(value: Any, limit: int = 200) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """Outcome of :meth:`Probe.run`, in three separate parts.

    * **Observed:** ``calls``, ``answer``, ``exception``, ``elapsed_ms``,
      ``deadline_exceeded``, ``still_running`` and ``scenarios`` (evidence).
    * **Asserted:** ``assertions``, filled by the ``assert_*`` methods.
    * **Inferred:** :attr:`retries` and :attr:`fallbacks`, heuristics over the
      call log, labelled as such.
    """

    input: Any
    answer: str | None
    raw_output: Any
    exception: BaseException | None
    elapsed_ms: float
    deadline_seconds: float | None
    deadline_exceeded: bool
    still_running: bool
    calls: list[CallRecord]
    scenarios: list[ScenarioEvidence]
    known_tools: list[str]
    assertions: list[AssertionRecord] = field(default_factory=list)

    # ---- inferred (heuristics) ------------------------------------------------

    @property
    def retries(self) -> dict[str, int]:
        """Heuristic: a call to the same tool directly after that tool raised."""
        out: dict[str, int] = {}
        for prev, cur in zip(self.calls, self.calls[1:]):
            if prev.outcome == "raised" and cur.tool == prev.tool:
                out[cur.tool] = out.get(cur.tool, 0) + 1
        return out

    @property
    def fallbacks(self) -> list[tuple[str, str]]:
        """Heuristic: ``(failed_tool, next_tool)`` when a different tool was called
        directly after a tool raised."""
        return [
            (prev.tool, cur.tool)
            for prev, cur in zip(self.calls, self.calls[1:])
            if prev.outcome == "raised" and cur.tool != prev.tool
        ]

    # ---- assertions -------------------------------------------------------------

    def _record(self, name: str, status: str, message: str, required: bool) -> None:
        self.assertions.append(AssertionRecord(name, status, message, required))
        if status != "pass" and required:
            raise AssertionError(f"{name}: {message}\n\n{self.summary()}")

    def _unassessable(self) -> str | None:
        if self.still_running:
            return "the agent was still running at the deadline, so there is no final answer"
        return None

    def assert_answer_contains(
        self, text: str, *, case_sensitive: bool = False, required: bool = True
    ) -> ProbeResult:
        """The final answer contains *text*."""
        name = f"answer contains {text!r}"
        if (why := self._unassessable()) is not None:
            self._record(name, "unassessed", why, required)
            return self
        if self.answer is None:
            reason = (
                f"no answer: the agent raised {type(self.exception).__name__}: {self.exception}"
                if self.exception is not None
                else "no answer was produced"
            )
            self._record(name, "fail", reason, required)
            return self
        hay, needle = (self.answer, text) if case_sensitive else (self.answer.lower(), text.lower())
        if needle in hay:
            self._record(name, "pass", "found", required)
        else:
            self._record(name, "fail", f"answer was {self.answer!r}", required)
        return self

    def assert_answer_not_contains(
        self, text: str, *, case_sensitive: bool = False, required: bool = True
    ) -> ProbeResult:
        """The final answer does not contain *text* (e.g. a success claim the agent
        could not have verified)."""
        name = f"answer does not contain {text!r}"
        if (why := self._unassessable()) is not None:
            self._record(name, "unassessed", why, required)
            return self
        answer = self.answer or ""
        hay, needle = (answer, text) if case_sensitive else (answer.lower(), text.lower())
        if needle in hay:
            self._record(name, "fail", f"answer was {self.answer!r}", required)
        else:
            self._record(name, "pass", "absent", required)
        return self

    def assert_no_crash(self, *, required: bool = True) -> ProbeResult:
        """No exception escaped the agent."""
        name = "no unhandled exception"
        if (why := self._unassessable()) is not None:
            self._record(name, "unassessed", why, required)
        elif self.exception is not None:
            self._record(
                name, "fail", f"{type(self.exception).__name__}: {self.exception}", required
            )
        else:
            self._record(name, "pass", "none raised", required)
        return self

    def assert_within(self, seconds: float, *, required: bool = True) -> ProbeResult:
        """The run finished within *seconds*. A run that is still going fails here,
        because the deadline was exceeded, which is an observed fact."""
        name = f"finished within {seconds:g}s"
        if self.still_running or self.elapsed_ms > seconds * 1000:
            self._record(
                name,
                "fail",
                f"took {self.elapsed_ms / 1000:.2f}s"
                + (" and is still running" if self.still_running else ""),
                required,
            )
        else:
            self._record(name, "pass", f"took {self.elapsed_ms / 1000:.2f}s", required)
        return self

    def assert_called(self, tool: str, *, required: bool = True) -> ProbeResult:
        name = f"tool {tool!r} was called"
        n = sum(1 for c in self.calls if c.tool == tool)
        self._record(name, "pass" if n else "fail", f"{n} call(s)", required)
        return self

    def assert_not_called(self, tool: str, *, required: bool = True) -> ProbeResult:
        name = f"tool {tool!r} was not called"
        n = sum(1 for c in self.calls if c.tool == tool)
        self._record(name, "fail" if n else "pass", f"{n} call(s)", required)
        return self

    def assert_scenarios_exercised(self) -> ProbeResult:
        """Every required scenario fired (also enforced when the probe closes)."""
        missing = [s for s in self.scenarios if s.required and s.status != "exercised"]
        if missing:
            self._record(
                "required scenarios exercised",
                "fail",
                "; ".join(_scenario_problem(s, self.known_tools, self.calls) for s in missing),
                True,
            )
        else:
            self._record("required scenarios exercised", "pass", "all fired", True)
        return self

    # ---- gate --------------------------------------------------------------------

    def gate(self) -> GateReport:
        """Pass only if every required scenario was exercised and every required
        assertion passed. Unassessed, error and missing evidence never pass."""
        failures: list[str] = []
        req_s = [s for s in self.scenarios if s.required]
        for s in req_s:
            if s.status != "exercised":
                failures.append(_scenario_problem(s, self.known_tools, self.calls))
        req_a = [a for a in self.assertions if a.required]
        for a in req_a:
            if a.status != "pass":
                failures.append(f"assertion {a.name!r} is {a.status}: {a.message}")
        return GateReport(
            passed=not failures,
            failures=failures,
            scenarios_exercised=sum(1 for s in req_s if s.status == "exercised"),
            scenarios_required=len(req_s),
            assertions_passed=sum(1 for a in req_a if a.status == "pass"),
            assertions_required=len(req_a),
        )

    # ---- presentation ----------------------------------------------------------

    def summary(self) -> str:
        lines = ["Observed"]
        if self.exception is not None:
            lines.append(f"  agent raised {type(self.exception).__name__}: {self.exception}")
        lines.append(f"  final answer: {self.answer!r}")
        timing = f"  elapsed: {self.elapsed_ms / 1000:.2f}s"
        if self.deadline_exceeded:
            timing += f" (deadline {self.deadline_seconds:g}s exceeded"
            timing += ", agent still running)" if self.still_running else ")"
        lines.append(timing)
        if not self.calls:
            lines.append("  tool calls: none")
        for c in self.calls:
            what = (
                f"returned {c.value}"
                if c.outcome == "returned"
                else f"raised {c.exception_type}: {c.exception_message}"
            )
            fault = f"  [injected: {c.fault}]" if c.injected else ""
            lines.append(f"  {c.index}. {c.tool}({_call_args(c)}) {what}{fault}")
        lines.append("  scenarios:")
        for s in self.scenarios:
            req = "required" if s.required else "optional"
            planned = "open-ended" if s.planned is None else str(s.planned)
            lines.append(
                f"    [{s.status}] {s.description} ({req}; injected {s.injected}/{planned}, "
                f"{s.tool_calls} call(s) to {s.tool})"
            )
        lines.append("Asserted")
        if not self.assertions:
            lines.append("  (no assertions yet)")
        for a in self.assertions:
            lines.append(f"  [{a.status}] {a.name}: {a.message}")
        lines.append("Inferred (heuristics over the call log)")
        lines.append(f"  retries: {self.retries or 'none'}")
        lines.append(f"  fallbacks: {self.fallbacks or 'none'}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "input": _short(self.input),
            "observed": {
                "answer": self.answer,
                "exception": None
                if self.exception is None
                else {"type": type(self.exception).__name__, "message": str(self.exception)},
                "elapsed_ms": round(self.elapsed_ms, 2),
                "deadline_seconds": self.deadline_seconds,
                "deadline_exceeded": self.deadline_exceeded,
                "still_running": self.still_running,
                "calls": [asdict(c) for c in self.calls],
                "scenarios": [asdict(s) for s in self.scenarios],
            },
            "asserted": [asdict(a) for a in self.assertions],
            "inferred": {
                "rule": "retry = same tool called right after it raised; "
                "fallback = different tool called right after a tool raised",
                "retries": self.retries,
                "fallbacks": [list(f) for f in self.fallbacks],
            },
            "gate": asdict(self.gate()),
        }


def _call_args(c: CallRecord) -> str:
    parts = [_short(a, 40) for a in c.args] + [f"{k}={_short(v, 40)}" for k, v in c.kwargs.items()]
    return ", ".join(parts)


def _scenario_problem(s: ScenarioEvidence, known_tools: list[str], calls: list[CallRecord]) -> str:
    if s.tool not in known_tools:
        return (
            f"scenario {s.description!r} not exercised: no wrapped tool is named {s.tool!r} "
            f"(wrapped: {', '.join(known_tools) or 'none'})"
        )
    if not s.boundary_reached:
        called = sorted({c.tool for c in calls})
        return (
            f"scenario {s.description!r} not exercised: the agent never called {s.tool!r} "
            f"(tools called: {', '.join(called) or 'none'}). It may have bypassed the tool "
            "or used a cached answer."
        )
    planned = "?" if s.planned is None else s.planned
    return (
        f"scenario {s.description!r} {s.status.replace('_', ' ')}: "
        f"{s.injected} of {planned} faults fired across {s.tool_calls} call(s)"
    )


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


class Probe:
    """Wraps tools, holds scenario rules and the call log, and enforces the gate.

    Use :func:`agentprobe.testing.probe` (or ``Probe()``) as a context manager.
    """

    def __init__(self, *, enforce: bool = True) -> None:
        self._lock = threading.RLock()
        self._rules: list[_Rule] = []
        self._calls: list[CallRecord] = []
        self._late_calls: list[CallRecord] = []
        self._per_tool: dict[str, int] = {}
        self._known_tools: list[str] = []
        self._active = False
        self._closed = False
        self._enforce = enforce
        self._results: list[ProbeResult] = []

    # ---- lifecycle --------------------------------------------------------------

    def __enter__(self) -> Probe:
        if self._closed:
            raise ProbeError("a probe cannot be reused after it has closed")
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
        if exc_type is None and self._enforce:
            report = self.gate()
            if not report.passed:
                raise ProbeGateError(str(report))

    async def __aenter__(self) -> Probe:
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.__exit__(exc_type, exc, tb)

    def close(self) -> None:
        """Stop injecting. Later calls through the wrappers pass straight through
        and are recorded in :attr:`late_calls`, never as evidence."""
        with self._lock:
            self._active = False
            self._closed = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def calls(self) -> list[CallRecord]:
        with self._lock:
            return list(self._calls)

    @property
    def late_calls(self) -> list[CallRecord]:
        with self._lock:
            return list(self._late_calls)

    # ---- scenarios ---------------------------------------------------------------

    def on(self, tool: Any) -> ToolScenarios:
        """Declare scenarios for *tool* (its name, or the tool object itself)."""
        return ToolScenarios(self, _tool_name(tool))

    def _add_rule(
        self,
        tool: str,
        description: str,
        required: bool,
        planned: int | None,
        decide: Callable[[int], _Action | None],
    ) -> None:
        with self._lock:
            if self._closed:
                raise ProbeError("cannot add scenarios to a closed probe")
            self._rules.append(
                _Rule(len(self._rules) + 1, tool, description, required, planned, decide)
            )

    def scenarios(self) -> list[ScenarioEvidence]:
        with self._lock:
            return [_evidence(r, self._per_tool.get(r.tool, 0)) for r in self._rules]

    # ---- wrapping ------------------------------------------------------------------

    def wrap(self, tools: Any) -> Any:
        """Return probed copies of *tools*. Accepts a single tool, a list/tuple, or a
        ``{name: callable}`` dict. Supports plain sync/async functions and LangChain
        ``BaseTool``s. The originals are never modified."""
        if isinstance(tools, dict):
            return {name: self._wrap_one(t, name) for name, t in tools.items()}
        if isinstance(tools, (list, tuple)):
            wrapped = [self._wrap_one(t) for t in tools]
            return type(tools)(wrapped) if isinstance(tools, tuple) else wrapped
        return self._wrap_one(tools)

    def _register(self, name: str) -> None:
        with self._lock:
            if name not in self._known_tools:
                self._known_tools.append(name)

    def _wrap_one(self, tool: Any, name: str | None = None) -> Any:
        try:
            from langchain_core.tools import BaseTool
        except ImportError:  # pragma: no cover
            BaseTool = None  # type: ignore[assignment,misc]  # noqa: N806

        if BaseTool is not None and isinstance(tool, BaseTool):
            return self._wrap_langchain(tool)
        if callable(tool):
            tool_name = name or getattr(tool, "__name__", None)
            if not tool_name:
                raise ProbeError(
                    f"cannot determine a name for {tool!r}; pass a dict {{name: tool}}"
                )
            self._register(tool_name)
            return self._wrap_callable(tool, tool_name)
        raise ProbeError(
            f"don't know how to wrap {type(tool).__name__}; pass a function or a LangChain tool"
        )

    def _wrap_callable(self, fn: Callable[..., Any], name: str) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await self._acall(name, fn, args, kwargs)

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._call(name, fn, args, kwargs)

        return wrapper

    def _wrap_langchain(self, tool: Any) -> Any:
        from langchain_core.tools import BaseTool, StructuredTool

        name = tool.name
        self._register(name)
        clone = tool.model_copy()
        inner_run = clone._run

        @functools.wraps(inner_run)
        def run(*args: Any, **kwargs: Any) -> Any:
            if _IN_WRAPPER.get():
                return inner_run(*args, **kwargs)
            return self._call(name, inner_run, args, kwargs)

        object.__setattr__(clone, "_run", run)

        has_own_async = (
            getattr(tool, "coroutine", None) is not None
            if isinstance(tool, StructuredTool)
            else type(tool)._arun is not BaseTool._arun
        )
        if has_own_async:
            inner_arun = clone._arun

            @functools.wraps(inner_arun)
            async def arun(*args: Any, **kwargs: Any) -> Any:
                if _IN_WRAPPER.get():
                    return await inner_arun(*args, **kwargs)
                return await self._acall(name, inner_arun, args, kwargs)

            object.__setattr__(clone, "_arun", arun)
        return clone

    # ---- the call path -------------------------------------------------------------

    def _begin(
        self, name: str, args: tuple, kwargs: dict
    ) -> tuple[CallRecord, _Action | None, bool]:
        with self._lock:
            if not self._active:
                rec = CallRecord(
                    index=len(self._late_calls) + 1,
                    tool=name,
                    tool_call_index=0,
                    args=list(args),
                    kwargs=_public_kwargs(kwargs),
                    outcome="returned",
                    after_probe_end=True,
                )
                return rec, None, True
            n = self._per_tool.get(name, 0) + 1
            self._per_tool[name] = n
            action: _Action | None = None
            rule_hit: _Rule | None = None
            for rule in self._rules:
                if rule.tool != name:
                    continue
                rule.offered += 1
                if (a := rule.decide(rule.offered)) is not None:
                    action, rule_hit = a, rule
                    rule.injected += 1
                    break
            rec = CallRecord(
                index=len(self._calls) + 1,
                tool=name,
                tool_call_index=n,
                args=list(args),
                kwargs=_public_kwargs(kwargs),
                outcome="returned",
                injected=action is not None,
                fault=rule_hit.description if rule_hit else None,
                rule_id=rule_hit.rule_id if rule_hit else None,
            )
            return rec, action, False

    def _finish(self, rec: CallRecord, late: bool) -> None:
        with self._lock:
            (self._late_calls if late else self._calls).append(rec)

    def _call(self, name: str, fn: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
        rec, action, late = self._begin(name, args, kwargs)
        start = time.perf_counter()
        token = _IN_WRAPPER.set(True)
        try:
            with _tracer.start_as_current_span(
                f"probe.tool.{name}",
                attributes={"tool.name": name, "failure.injected": action is not None},
            ) as span:
                rec.span_id = format(span.get_span_context().span_id, "016x")
                if action is not None and action.kind == "raise":
                    assert action.exc is not None
                    raise action.exc
                if action is not None and action.kind == "return":
                    result = action.value
                else:
                    if action is not None and action.kind == "delay":
                        time.sleep(action.seconds)
                    result = fn(*args, **kwargs)
                rec.value = _short(result)
                return result
        except BaseException as exc:
            rec.outcome = "raised"
            rec.exception_type = type(exc).__name__
            rec.exception_message = str(exc)[:300]
            raise
        finally:
            _IN_WRAPPER.reset(token)
            rec.duration_ms = (time.perf_counter() - start) * 1000
            self._finish(rec, late)

    async def _acall(self, name: str, fn: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
        rec, action, late = self._begin(name, args, kwargs)
        start = time.perf_counter()
        token = _IN_WRAPPER.set(True)
        try:
            with _tracer.start_as_current_span(
                f"probe.tool.{name}",
                attributes={"tool.name": name, "failure.injected": action is not None},
            ) as span:
                rec.span_id = format(span.get_span_context().span_id, "016x")
                if action is not None and action.kind == "raise":
                    assert action.exc is not None
                    raise action.exc
                if action is not None and action.kind == "return":
                    result = action.value
                else:
                    if action is not None and action.kind == "delay":
                        await asyncio.sleep(action.seconds)
                    result = await fn(*args, **kwargs)
                rec.value = _short(result)
                return result
        except BaseException as exc:
            rec.outcome = "raised"
            rec.exception_type = type(exc).__name__
            rec.exception_message = str(exc)[:300]
            raise
        finally:
            _IN_WRAPPER.reset(token)
            rec.duration_ms = (time.perf_counter() - start) * 1000
            self._finish(rec, late)

    # ---- running the agent ---------------------------------------------------------

    def run(
        self, agent: Any, input: Any = None, *args: Any, timeout: float | None = None, **kwargs: Any
    ) -> ProbeResult:
        """Run *agent* on *input* and return a :class:`ProbeResult`.

        *agent* may be a function (sync or async) or an object with ``invoke`` /
        ``ainvoke`` (LangChain runnables, compiled LangGraph graphs, AgentProbe
        adapters). Exceptions from the agent are captured in the result, not raised,
        so you can assert on them.

        With ``timeout`` the agent runs in a worker thread. If it is still running at
        the deadline, the result is marked ``deadline_exceeded`` and ``still_running``,
        and the probe closes so late tool calls cannot count as evidence.
        """
        if self._closed:
            raise ProbeError("this probe is closed; use a new probe for a new test")
        call = _agent_callable(agent)
        call_args = ((input,) if input is not None else ()) + args
        start = time.perf_counter()
        output: Any = None
        error: BaseException | None = None
        still_running = False
        deadline_exceeded = False

        def target() -> Any:
            res = call(*call_args, **kwargs)
            if inspect.isawaitable(res):
                return asyncio.run(_await(res))
            return res

        if timeout is None:
            try:
                output = target()
            except Exception as exc:  # noqa: BLE001 - captured for assertions
                error = exc
        else:
            ctx = contextvars.copy_context()
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = pool.submit(ctx.run, target)
            pool.shutdown(wait=False)
            try:
                output = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                deadline_exceeded = True
                still_running = not future.done()
                if still_running:
                    self.close()
                else:  # finished in the tiny gap after the deadline
                    try:
                        output = future.result()
                    except Exception as exc:  # noqa: BLE001
                        error = exc
            except Exception as exc:  # noqa: BLE001
                error = exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        if timeout is not None and not deadline_exceeded and elapsed_ms > timeout * 1000:
            deadline_exceeded = True
        answer = None if (error is not None or still_running) else _answer_text(output)
        result = ProbeResult(
            input=input,
            answer=answer,
            raw_output=output,
            exception=error,
            elapsed_ms=elapsed_ms,
            deadline_seconds=timeout,
            deadline_exceeded=deadline_exceeded,
            still_running=still_running,
            calls=self.calls,
            scenarios=self.scenarios(),
            known_tools=list(self._known_tools),
        )
        with self._lock:
            self._results.append(result)
        return result

    # ---- gate ------------------------------------------------------------------------

    def gate(self) -> GateReport:
        """The gate over the whole probe: every required scenario exercised (from all
        calls made while the probe was active) and every required assertion passed
        in every result."""
        with self._lock:
            results = list(self._results)
        merged = ProbeResult(
            input=None,
            answer=None,
            raw_output=None,
            exception=None,
            elapsed_ms=0.0,
            deadline_seconds=None,
            deadline_exceeded=False,
            still_running=False,
            calls=self.calls,
            scenarios=self.scenarios(),
            known_tools=list(self._known_tools),
            assertions=[a for r in results for a in r.assertions],
        )
        return merged.gate()


async def _await(aw: Any) -> Any:
    return await aw


def _agent_callable(agent: Any) -> Callable[..., Any]:
    if callable(agent) and not hasattr(agent, "invoke"):
        return agent  # type: ignore[no-any-return]
    if hasattr(agent, "invoke"):
        return agent.invoke  # type: ignore[no-any-return]
    if hasattr(agent, "ainvoke"):
        return agent.ainvoke  # type: ignore[no-any-return]
    raise ProbeError(f"{type(agent).__name__} is not callable and has no invoke()")


def _answer_text(output: Any) -> str | None:
    if output is None:
        return None
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        if "output" in output:
            return _answer_text(output["output"])
        if output.get("messages"):
            return message_text(output["messages"][-1])
        return str(output)
    if hasattr(output, "content"):
        return message_text(output)
    return str(output)


def _tool_name(tool: Any) -> str:
    if isinstance(tool, str):
        return tool
    name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
    if not name:
        raise ProbeError(f"cannot determine the tool name of {tool!r}")
    return str(name)


def _public_kwargs(kwargs: dict) -> dict:
    """Drop framework plumbing (callbacks, run managers, config) from the log."""
    return {k: v for k, v in kwargs.items() if k not in {"run_manager", "callbacks", "config"}}


__all__ = [
    "SCHEMA_VERSION",
    "Probe",
    "ProbeResult",
    "ProbeError",
    "ProbeGateError",
    "ToolScenarios",
    "CallRecord",
    "ScenarioEvidence",
    "AssertionRecord",
    "GateReport",
]
