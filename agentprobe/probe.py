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

Async agents and async tests use ``await p.arun(...)``.

Design contract (see ``docs/instrumentation.md``):

* **Your tools are never modified.** :meth:`Probe.wrap` returns new objects (a new
  function, or a shallow copy of a LangChain tool). Originals stay as they were,
  so two probes on the same tool cannot interfere.
* **Evidence, not hope.** Each scenario records whether its tool was reached and
  whether its fault was injected. A required scenario that never fired is
  ``not_exercised`` and fails the gate.
* **Calls are recorded when they start.** Every tool call appears in the log the
  moment it begins (``status="pending"``) and is completed when it returns or
  raises, so a call still in flight at a deadline is never invisible.
* **Observed, asserted and inferred are separate.** :class:`ProbeResult` keeps the
  call log, your assertion results (``pass``/``fail``/``unassessed``/``error``)
  and heuristic labels (retry, fallback) apart. Answer assertions need an actual
  answer: a crash or ``None`` never satisfies them.
* **Still-running agents are reported, not hidden.** A result is a frozen snapshot
  taken at the end of the run. If a sync agent outlives its deadline the result
  says so, the probe closes, calls that complete later are marked
  ``completed_after_close``, and calls that *start* later pass through without a
  fault as ``after_probe_end``. Neither changes an earlier verdict.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import copy
import dataclasses
import datetime as _dt
import decimal
import enum
import functools
import hashlib
import inspect
import math
import pathlib
import random
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentprobe._otel import get_tracer
from agentprobe._text import message_text
from agentprobe.faults import (
    ToolCallAPIError,
    ToolCallError,
    ToolCallMalformedOutput,
    ToolCallRateLimited,
    ToolCallTimeout,
)

SCHEMA_VERSION = "1.2"

_tracer = get_tracer(__name__)

#: Wrapper keys currently executing in this context. A wrapped LangChain tool's
#: async method may delegate to its own wrapped sync method; that delegation must
#: not count twice. Keyed by wrapper identity, so a *different* wrapped tool
#: called from inside a tool is still recorded and still gets its faults.
_ACTIVE_WRAPPERS: contextvars.ContextVar[frozenset[int]] = contextvars.ContextVar(
    "agentprobe_active_wrappers", default=frozenset()
)


class ProbeError(Exception):
    """Misuse of the probe API (bad tool, bad scenario, wrong lifecycle)."""


class ProbeGateError(AssertionError):
    """A required scenario was not exercised or a required assertion did not pass."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _count(name: str, value: Any, *, allow_none: bool) -> int | None:
    """A positive whole number of calls, or ``None`` (every call) where allowed.
    Zero, negatives, fractions, booleans and non-finite values are rejected: a
    rule that can never inject must not be able to count as exercised."""
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        allowed = "a positive whole number or None" if allow_none else "a positive whole number"
        raise ProbeError(f"{name} must be {allowed}, got {value!r}")
    return value


def _finite_number(name: str, value: Any, *, low: float, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProbeError(f"{name} must be a finite number, got {value!r}")
    if value < low or (high is not None and value > high):
        bound = f"between {low:g} and {high:g}" if high is not None else f">= {low:g}"
        raise ProbeError(f"{name} must be {bound}, got {value!r}")
    return float(value)


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


def _fault_factory(
    with_: str | type[BaseException] | BaseException,
) -> Callable[[], BaseException]:
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


def _first_calls(n: int) -> str:
    return f"the first {n} call{'s' if n > 1 else ''}"


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
    consumes_every_call: bool  # True if no later rule for this tool can ever be reached
    injected: int = 0
    offered: int = 0  # calls that reached this rule (not consumed by earlier rules)


class ToolScenarios:
    """Builder returned by :meth:`Probe.on`. Each method adds one scenario rule.

    All rules are deterministic except :meth:`with_probability`. Rules for the
    same tool form a sequence: each call is offered to the rules in declaration
    order, and each rule counts only the calls that reach it. So
    ``fail_first(1).returns("stale", times=2)`` fails call 1, returns stale data
    for calls 2 and 3, and lets call 4 through. A rule that applies to every call
    (``always_fail()``, or ``returns``/``slow`` with ``times=None``) must be the
    last rule for its tool; adding one after it raises :class:`ProbeError`.
    """

    def __init__(self, probe: Probe, tool: str) -> None:
        self._probe = probe
        self._tool = tool

    def fail_first(
        self, n: int = 1, *, with_: Any = "timeout", required: bool = True
    ) -> ToolScenarios:
        """Fail the first *n* calls to this tool, then let calls through."""
        count = _count("fail_first(n)", n, allow_none=False)
        assert count is not None
        make = _fault_factory(with_)
        self._probe._add_rule(
            self._tool,
            f"{self._tool}: fail {_first_calls(count)} with {_fault_label(with_)}",
            required,
            count,
            lambda i: _Action("raise", exc=make()) if i <= count else None,
            consumes_every_call=False,
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
            consumes_every_call=True,
        )
        return self

    def slow(self, *, ms: float, times: int | None = 1, required: bool = True) -> ToolScenarios:
        """Delay the first *times* calls by *ms* milliseconds, then run them normally.
        ``times=None`` delays every call."""
        seconds = _finite_number("slow(ms)", ms, low=0.0) / 1000.0
        count = _count("slow(times)", times, allow_none=True)
        label = "every call" if count is None else _first_calls(count)
        self._probe._add_rule(
            self._tool,
            f"{self._tool}: delay {label} by {ms:g} ms",
            required,
            count,
            lambda i: _Action("delay", seconds=seconds) if count is None or i <= count else None,
            consumes_every_call=count is None,
        )
        return self

    def returns(self, value: Any, *, times: int | None = 1, required: bool = True) -> ToolScenarios:
        """Return *value* instead of calling the tool for the first *times* calls
        (``None`` = every call). Use it for wrong, stale or incomplete data."""
        count = _count("returns(times)", times, allow_none=True)
        label = "every call" if count is None else _first_calls(count)
        self._probe._add_rule(
            self._tool,
            f"{self._tool}: return {_short(value, 60)} for {label}",
            required,
            count,
            lambda i: (
                _Action("return", value=copy.deepcopy(value))
                if count is None or i <= count
                else None
            ),
            consumes_every_call=count is None,
        )
        return self

    def with_probability(
        self, p: float, *, with_: Any = "timeout", seed: int | None = None, required: bool = True
    ) -> ToolScenarios:
        """Fail each call with probability *p* (seeded). For experiments; regression
        tests should prefer the deterministic rules. Exercised once at least one
        fault fired."""
        prob = _finite_number("probability", p, low=0.0, high=1.0)
        rng = random.Random(seed)
        make = _fault_factory(with_)
        self._probe._add_rule(
            self._tool,
            f"{self._tool}: fail with {_fault_label(with_)} at p={prob:g}"
            + (f" (seed {seed})" if seed is not None else ""),
            required,
            None,
            lambda i: _Action("raise", exc=make()) if rng.random() < prob else None,
            consumes_every_call=False,
        )
        return self


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
    """One tool call seen through a wrapper. Observed, never inferred.

    Created when the call **starts** (``status="pending"``), completed when it
    returns or raises. ``started_seq`` / ``completed_seq`` order start and
    completion events across all tools in the probe.
    """

    index: int  # invocation order across all tools in this probe (1-based)
    tool: str
    tool_call_index: int  # nth call to this tool while the probe was active; 0 if after end
    args: list[Any]  # JSON-safe snapshot taken when the call started; never changes
    kwargs: dict[str, Any]  # JSON-safe snapshot taken when the call started
    status: str = "pending"  # "pending" | "completed"
    outcome: str | None = None  # "returned" | "raised" once completed
    value: str | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    injected: bool = False
    fault: str | None = None  # scenario description when injected
    rule_id: int | None = None
    duration_ms: float | None = None
    span_id: str | None = None
    started_seq: int = 0
    completed_seq: int | None = None
    after_probe_end: bool = False  # started after the probe closed (never evidence)
    completed_after_close: bool = False  # started while active, finished after close
    #: The live objects your agent passed, kept for debugging only. They are the
    #: caller's objects and may change after the call; evidence and reports use
    #: the snapshots in ``args``/``kwargs``. Never serialized.
    raw_args: tuple[Any, ...] = field(default=(), repr=False, compare=False)
    raw_kwargs: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    #: Exact, immutable copies of the arguments taken when the call started, used
    #: by ``assert_called_with``. Unlike ``args``/``kwargs`` (a report snapshot that
    #: shortens bytes and long reprs), they keep types and full contents. Values
    #: that can't be copied exactly are opaque and never compare equal. Never
    #: serialized.
    exact_args: tuple[Any, ...] = field(default=(), repr=False, compare=False)
    exact_kwargs: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    #: The tool's signature, when known, so positional and keyword forms of the
    #: same argument match. Never serialized.
    signature: inspect.Signature | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in _UNSERIALIZED_CALL_FIELDS
        }


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
    try:
        text = repr(value)
    except Exception as exc:  # noqa: BLE001 - repr of arbitrary user objects
        text = f"<unrepresentable {type(value).__name__}: {type(exc).__name__}>"
    return text if len(text) <= limit else text[: limit - 3] + "..."


_UNSERIALIZED_CALL_FIELDS = frozenset(
    {"raw_args", "raw_kwargs", "exact_args", "exact_kwargs", "signature"}
)


class _Opaque:
    """Stands in for a value that can't be copied exactly. Equal only to itself,
    so a recorded opaque value never matches an expectation."""

    __slots__ = ("type_name",)

    def __init__(self, value: Any) -> None:
        self.type_name = f"{type(value).__module__}.{type(value).__qualname__}"

    def __repr__(self) -> str:
        return f"<opaque {self.type_name}>"


#: Immutable types compared as they are, with Python's own equality. Matched by
#: exact type: a subclass (a pandas ``Timestamp`` is a ``datetime``) may define
#: its own equality, so it is opaque.
_EXACT_SCALARS = frozenset(
    {
        type(None),
        bool,
        int,
        float,
        complex,
        str,
        _dt.datetime,
        _dt.date,
        _dt.time,
        _dt.timedelta,
        decimal.Decimal,
        uuid.UUID,
    }
)
_EXACT_PATHS = frozenset(
    {
        pathlib.PurePath,
        pathlib.PurePosixPath,
        pathlib.PureWindowsPath,
        pathlib.Path,
        pathlib.PosixPath,
        pathlib.WindowsPath,
    }
)


def _exact(value: Any, _depth: int = 0) -> Any:
    """An immutable copy of *value* that compares like the original did when copied.

    Only built-in types are supported, matched by exact type: ``None``, ``bool``,
    numbers, ``str``, dates and times, ``Decimal``, ``UUID``, paths, enum members,
    bytes, and ``list``/``tuple``/``dict``/``set``/``frozenset`` of these.
    Containers become tagged tuples, so a list never equals a tuple and dict keys
    keep their types, as with Python's ``==``. Bytes keep their length and a
    SHA-256 of the full content. Everything else, including subclasses of these
    types, dataclasses and pydantic models, becomes ``_Opaque``: their equality is
    user-defined, so a copy can't be trusted to compare the same way.
    """
    if _depth > 50:
        return _Opaque(value)
    kind = type(value)
    if kind in _EXACT_SCALARS or kind in _EXACT_PATHS:
        return value
    if isinstance(value, enum.Enum):
        return value  # members are singletons: keep the member itself, not a copy
    if kind in (bytes, bytearray, memoryview):
        raw = bytes(value)
        return ("bytes", len(raw), hashlib.sha256(raw).digest())
    if kind is tuple:
        return ("tuple", tuple(_exact(v, _depth + 1) for v in value))
    if kind is list:
        return ("list", tuple(_exact(v, _depth + 1) for v in value))
    if kind in (set, frozenset):
        return ("set", frozenset(_exact(v, _depth + 1) for v in value))
    if kind is dict:
        return (
            "dict",
            frozenset((_exact(k, _depth + 1), _exact(v, _depth + 1)) for k, v in value.items()),
        )
    return _Opaque(value)


def _exact_or_opaque(value: Any) -> Any:
    try:
        return _exact(value)
    except Exception:  # noqa: BLE001 - e.g. a dict key that is unhashable after copying
        return _Opaque(value)


def _find_opaque(frozen: Any) -> _Opaque | None:
    """The first opaque value inside a copy made by ``_exact``, if any."""
    if isinstance(frozen, _Opaque):
        return frozen
    if isinstance(frozen, (tuple, frozenset)):
        for v in frozen:
            if (found := _find_opaque(v)) is not None:
                return found
    return None


def _has_opaque(frozen: Any) -> bool:
    return _find_opaque(frozen) is not None


def _signature_of(tool: Any) -> inspect.Signature | None:
    """Parameters of a wrapped tool, so ``f("42")`` and ``f(order_id="42")`` match."""
    try:
        from langchain_core.tools import BaseTool
    except ImportError:  # pragma: no cover
        BaseTool = None  # type: ignore[assignment,misc]  # noqa: N806
    try:
        if BaseTool is not None and isinstance(tool, BaseTool):
            names = list(tool.args)
            return inspect.Signature(
                [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in names]
            )
        return inspect.signature(tool)
    except Exception:  # noqa: BLE001 - builtins, odd callables, schema errors
        return None


def _normalize_call(
    sig: inspect.Signature | None, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    """Positional values that have no name, and every named value, per *sig*.

    Without a signature the call is returned as given. Returns ``None`` when the
    arguments don't fit the signature.
    """
    if sig is None:
        return tuple(args), dict(kwargs)
    try:
        bound = sig.bind_partial(*args, **kwargs)
    except TypeError:
        return None
    positional: list[Any] = []
    named: dict[str, Any] = {}
    for pname, value in bound.arguments.items():
        kind = sig.parameters[pname].kind
        if kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        elif kind is inspect.Parameter.VAR_POSITIONAL:
            positional.extend(value)
        elif kind is inspect.Parameter.VAR_KEYWORD:
            named.update(value)
        else:
            named[pname] = value
    return tuple(positional), named


def _match_call(call: CallRecord, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool | None:
    """True if *call* matches, False if it doesn't, None if it can't be decided."""
    actual = _normalize_call(call.signature, call.exact_args, call.exact_kwargs)
    expected = _normalize_call(call.signature, args, kwargs)
    if actual is None or expected is None:
        actual = (tuple(call.exact_args), dict(call.exact_kwargs))
        expected = (tuple(args), dict(kwargs))
    (a_pos, a_named), (e_pos, e_named) = actual, expected
    undecided = False
    if len(a_pos) != len(e_pos):
        return False
    pairs = list(zip(a_pos, e_pos))
    for k, v in e_named.items():
        if k not in a_named:
            return False
        pairs.append((a_named[k], v))
    for a, e in pairs:
        if a != e:
            if _has_opaque(a):
                undecided = True
            else:
                return False
    return None if undecided else True


def _jsonable(value: Any, _depth: int = 0) -> Any:
    """Convert to JSON-safe data without touching the caller's objects.

    JSON-native values pass through. Datetimes, dates, times, decimals, UUIDs,
    paths and enums become strings. Bytes become ``{"type": "bytes", "length",
    "preview"}``. Anything else becomes ``{"type": <qualified name>, "repr": ...}``.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if _depth > 20:
        return {"type": type(value).__name__, "repr": _short(value)}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v, _depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, enum.Enum):
        return _jsonable(value.value, _depth + 1)
    if isinstance(value, (decimal.Decimal, uuid.UUID, pathlib.PurePath)):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {"type": "bytes", "length": len(raw), "preview": raw[:32].hex()}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "repr": _short(value)}


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """Outcome of :meth:`Probe.run` / :meth:`Probe.arun`, in three separate parts.

    * **Observed:** ``calls`` (a snapshot at the end of the run, pending calls
      included), ``answer``, ``exception``, ``elapsed_ms``, ``deadline_exceeded``,
      ``still_running``, ``cancelled`` and ``scenarios`` (evidence).
    * **Asserted:** ``assertions``, filled by the ``assert_*`` methods.
    * **Inferred:** :attr:`retries` and :attr:`fallbacks`, heuristics over the
      call log, labelled as such.

    The snapshot never changes after the run returns, even if a still-running
    agent completes more calls later.
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
    cancelled: bool = False  # cancellation confirmed (async agents only)
    cancel_requested: bool = False  # cancellation was requested at the deadline
    assertions: list[AssertionRecord] = field(default_factory=list)

    # ---- inferred (heuristics) ------------------------------------------------

    def _ordered_pairs(self) -> list[tuple[CallRecord, CallRecord]]:
        """Consecutive calls in invocation order where the first had already
        failed before the second started (concurrent calls are never paired)."""
        ordered = sorted(self.calls, key=lambda c: c.index)
        return [
            (prev, cur)
            for prev, cur in zip(ordered, ordered[1:])
            if prev.outcome == "raised"
            and prev.completed_seq is not None
            and prev.completed_seq < cur.started_seq
        ]

    @property
    def retries(self) -> dict[str, int]:
        """Heuristic: the same tool called next, after that tool had raised."""
        out: dict[str, int] = {}
        for prev, cur in self._ordered_pairs():
            if cur.tool == prev.tool:
                out[cur.tool] = out.get(cur.tool, 0) + 1
        return out

    @property
    def fallbacks(self) -> list[tuple[str, str]]:
        """Heuristic: ``(failed_tool, next_tool)`` when a different tool was called
        next, after a tool had raised."""
        return [(p.tool, c.tool) for p, c in self._ordered_pairs() if c.tool != p.tool]

    # ---- assertions -------------------------------------------------------------

    def _record(self, name: str, status: str, message: str, required: bool) -> None:
        self.assertions.append(AssertionRecord(name, status, message, required))
        if status != "pass" and required:
            raise AssertionError(f"{name}: {message}\n\n{self.summary()}")

    def _no_answer_reason(self) -> tuple[str, str] | None:
        """Why an answer assertion cannot pass: (status, message), or None."""
        if self.still_running:
            return (
                "unassessed",
                "the agent was still running at the deadline, so there is no final answer",
            )
        if self.cancelled:
            return ("unassessed", "the agent was cancelled at the deadline; no final answer")
        if self.exception is not None:
            return (
                "fail",
                f"no answer: the agent raised {type(self.exception).__name__}: {self.exception}",
            )
        if self.answer is None:
            return ("fail", "no answer: the agent returned None")
        return None

    def assert_answer_contains(
        self, text: str, *, case_sensitive: bool = False, required: bool = True
    ) -> ProbeResult:
        """The agent produced a final answer, and it contains *text*."""
        name = f"answer contains {text!r}"
        if (reason := self._no_answer_reason()) is not None:
            self._record(name, *reason, required)
            return self
        assert self.answer is not None
        hay, needle = (self.answer, text) if case_sensitive else (self.answer.lower(), text.lower())
        if needle in hay:
            self._record(name, "pass", "found", required)
        else:
            self._record(name, "fail", f"answer was {self.answer!r}", required)
        return self

    def assert_answer_not_contains(
        self, text: str, *, case_sensitive: bool = False, required: bool = True
    ) -> ProbeResult:
        """The agent produced a final answer, and it does not contain *text*.

        A crash or a ``None`` answer does **not** pass: "no answer" is not the
        same as "an answer without the claim". Check crashes with
        :meth:`assert_no_crash`.
        """
        name = f"answer does not contain {text!r}"
        if (reason := self._no_answer_reason()) is not None:
            self._record(name, *reason, required)
            return self
        assert self.answer is not None
        hay, needle = (self.answer, text) if case_sensitive else (self.answer.lower(), text.lower())
        if needle in hay:
            self._record(name, "fail", f"answer was {self.answer!r}", required)
        else:
            self._record(name, "pass", "absent", required)
        return self

    def assert_no_crash(self, *, required: bool = True) -> ProbeResult:
        """No exception escaped the agent."""
        name = "no unhandled exception"
        if self.still_running or self.cancelled:
            self._record(name, "unassessed", "the agent did not finish", required)
        elif self.exception is not None:
            self._record(
                name, "fail", f"{type(self.exception).__name__}: {self.exception}", required
            )
        else:
            self._record(name, "pass", "none raised", required)
        return self

    def assert_within(self, seconds: float, *, required: bool = True) -> ProbeResult:
        """The run finished within *seconds*. A run that is still going or was
        cancelled fails here: exceeding the deadline is an observed fact."""
        name = f"finished within {seconds:g}s"
        took = f"took {self.elapsed_ms / 1000:.2f}s"
        if self.still_running:
            self._record(name, "fail", took + " and is still running", required)
        elif self.cancelled:
            self._record(name, "fail", took + " and was cancelled", required)
        elif self.elapsed_ms > seconds * 1000:
            self._record(name, "fail", took, required)
        else:
            self._record(name, "pass", took, required)
        return self

    def _calls_to(self, tool: str) -> list[CallRecord]:
        return [c for c in self.calls if c.tool == tool and not c.after_probe_end]

    def assert_called(self, tool: str, *, required: bool = True) -> ProbeResult:
        """*tool* was called (a call in flight counts: the boundary was reached)."""
        name = f"tool {tool!r} was called"
        n = len(self._calls_to(tool))
        if n:
            self._record(name, "pass", f"{n} call(s)", required)
        elif self.still_running:
            self._record(
                name, "unassessed", "not yet called, and the agent is still running", required
            )
        else:
            self._record(name, "fail", "0 calls", required)
        return self

    def assert_not_called(self, tool: str, *, required: bool = True) -> ProbeResult:
        """*tool* was never called. A call in flight is a call. While the agent is
        still running, "not called yet" cannot establish "never called"."""
        name = f"tool {tool!r} was not called"
        calls = self._calls_to(tool)
        if calls:
            pending = sum(1 for c in calls if c.status == "pending")
            detail = f"{len(calls)} call(s)" + (f", {pending} still in flight" if pending else "")
            self._record(name, "fail", detail, required)
        elif self.still_running:
            self._record(
                name, "unassessed", "not called yet, but the agent is still running", required
            )
        else:
            self._record(name, "pass", "0 calls", required)
        return self

    def _record_call_check(
        self, name: str, status: str, message: str, calls: list[CallRecord], required: bool
    ) -> ProbeResult:
        if status == "unassessed":
            message += ", and the agent is still running"
        if status != "pass":
            recent = ", ".join(f"#{c.index} {c.tool}({_call_args(c)})" for c in calls[-5:])
            message += f". Recent calls: {recent or 'none'}"
        self._record(name, status, message, required)
        return self

    def assert_called_with(self, tool: str, /, *args: Any, **kwargs: Any) -> ProbeResult:
        """Some call to *tool* had these arguments.

        Unlike ``unittest.mock``'s ``assert_called_with``, any call can match (not
        only the last), and arguments you leave out aren't checked. When the tool's
        signature is known, positional and keyword forms of an argument match each
        other; extra positional arguments (``*args``) must match exactly. Values are
        compared exactly, with Python's ``==`` on copies taken when the call
        started. An argument that couldn't be copied exactly (an arbitrary object)
        leaves the assertion ``unassessed``, never passed.

        All keywords describe the tool call, including keywords named ``tool`` or
        ``required``. The assertion is required.
        """
        expected_args = tuple(_exact_or_opaque(v) for v in args)
        expected_kwargs = {k: _exact_or_opaque(v) for k, v in kwargs.items()}
        for v in (*expected_args, *expected_kwargs.values()):
            if (opaque := _find_opaque(v)) is not None:
                raise ProbeError(
                    f"assert_called_with can't compare {opaque.type_name} exactly; expected "
                    "values must be built-in: None, bool, numbers, str, bytes, dates and "
                    "times, Decimal, UUID, paths, enum members, or list/tuple/dict/set of "
                    "these. Check other objects on result.calls[i].raw_args yourself"
                )
        calls = self._calls_to(tool)
        name = (
            f"tool {tool!r} called with args={_short(_jsonable(list(args)))}, "
            f"kwargs={_short(_jsonable(kwargs))}"
        )
        undecided: list[CallRecord] = []
        for call in calls:
            verdict = _match_call(call, expected_args, expected_kwargs)
            if verdict:
                self._record(name, "pass", f"matched call #{call.index}", True)
                return self
            if verdict is None:
                undecided.append(call)
        if undecided:
            ids = ", ".join(f"#{c.index}" for c in undecided)
            message = f"call(s) {ids} had arguments that can't be compared exactly"
            self._record(name, "unassessed", message, True)
            return self
        status = "unassessed" if self.still_running else "fail"
        return self._record_call_check(name, status, "no matching invocation", calls, True)

    def assert_call_order(self, tools: Sequence[str], *, required: bool = True) -> ProbeResult:
        """Each tool was invoked after the preceding one, with other calls allowed between."""
        if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
            raise ProbeError("call order must be a sequence of tool names")
        if not all(isinstance(tool, str) for tool in tools):
            raise ProbeError("call order must contain only tool names")
        name = f"tool call order {list(tools)!r}"
        selected = set(tools)
        calls = sorted(
            (c for c in self.calls if not c.after_probe_end and c.tool in selected),
            key=lambda c: c.index,
        )
        remaining = iter(calls)
        for position, tool in enumerate(tools, start=1):
            if not any(c.tool == tool for c in remaining):
                status = "unassessed" if self.still_running else "fail"
                message = f"did not observe {tool!r} at position {position} in the expected order"
                return self._record_call_check(name, status, message, calls, required)
        self._record(name, "pass", "observed in invocation order", required)
        return self

    def assert_called_times(self, tool: str, n: int, *, required: bool = True) -> ProbeResult:
        """Exactly *n* invocations of *tool* started."""
        return self._assert_call_count(tool, n, at_most=False, required=required)

    def assert_called_at_most(self, tool: str, n: int, *, required: bool = True) -> ProbeResult:
        """At most *n* invocations of *tool* started."""
        return self._assert_call_count(tool, n, at_most=True, required=required)

    def _assert_call_count(
        self, tool: str, n: int, *, at_most: bool, required: bool
    ) -> ProbeResult:
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ProbeError(f"call count must be a non-negative integer, got {n!r}")
        name = f"tool {tool!r} called {'at most' if at_most else 'exactly'} {n} time(s)"
        calls = self._calls_to(tool)
        count = len(calls)
        if count > n:
            status = "fail"
        elif self.still_running:
            status = "unassessed"
        elif at_most or count == n:
            status = "pass"
        else:
            status = "fail"
        return self._record_call_check(name, status, f"observed {count} call(s)", calls, required)

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
        return _gate(self.scenarios, self.assertions, self.known_tools, self.calls)

    # ---- presentation ----------------------------------------------------------

    def summary(self) -> str:
        lines = ["Observed"]
        if self.exception is not None:
            lines.append(f"  agent raised {type(self.exception).__name__}: {self.exception}")
        lines.append(f"  final answer: {self.answer!r}")
        timing = f"  elapsed: {self.elapsed_ms / 1000:.2f}s"
        if self.deadline_exceeded:
            timing += f" (deadline {self.deadline_seconds:g}s exceeded"
            if self.still_running:
                timing += ", agent still running"
                if self.cancel_requested:
                    timing += " after cancellation was requested"
            elif self.cancelled:
                timing += ", agent cancelled"
            timing += ")"
        lines.append(timing)
        if not self.calls:
            lines.append("  tool calls: none")
        for c in sorted(self.calls, key=lambda c: c.index):
            if c.status == "pending":
                what = "IN FLIGHT (no result yet)"
            elif c.outcome == "returned":
                what = f"returned {c.value}"
            else:
                what = f"raised {c.exception_type}: {c.exception_message}"
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
        """JSON-safe representation (``json.dumps`` always works). Non-JSON values
        are converted as described in :func:`_jsonable`; caller objects are not
        modified."""
        return {
            "schema_version": SCHEMA_VERSION,
            "input": _jsonable(self.input),
            "observed": {
                "answer": self.answer,
                "exception": None
                if self.exception is None
                else {"type": type(self.exception).__name__, "message": str(self.exception)},
                "elapsed_ms": round(self.elapsed_ms, 2),
                "deadline_seconds": self.deadline_seconds,
                "deadline_exceeded": self.deadline_exceeded,
                "still_running": self.still_running,
                "cancel_requested": self.cancel_requested,
                "cancelled": self.cancelled,
                "calls": [c.to_dict() for c in sorted(self.calls, key=lambda c: c.index)],
                "scenarios": [dataclasses.asdict(s) for s in self.scenarios],
            },
            "asserted": [dataclasses.asdict(a) for a in self.assertions],
            "inferred": {
                "rule": "retry = same tool called next after it had raised; "
                "fallback = different tool called next after a tool had raised; "
                "calls that overlapped in time are never paired",
                "retries": self.retries,
                "fallbacks": [list(f) for f in self.fallbacks],
            },
            "gate": dataclasses.asdict(self.gate()),
        }


def _gate(
    scenarios: list[ScenarioEvidence],
    assertions: list[AssertionRecord],
    known_tools: list[str],
    calls: list[CallRecord],
) -> GateReport:
    failures: list[str] = []
    req_s = [s for s in scenarios if s.required]
    for s in req_s:
        if s.status != "exercised":
            failures.append(_scenario_problem(s, known_tools, calls))
    req_a = [a for a in assertions if a.required]
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


def _call_args(c: CallRecord) -> str:
    """Render the frozen argument snapshot (never the live objects)."""

    def show(v: Any) -> str:
        if isinstance(v, dict) and set(v) == {"type", "repr"}:
            return str(v["repr"])[:40]
        return _short(v, 40)

    parts = [show(a) for a in c.args] + [f"{k}={show(v)}" for k, v in c.kwargs.items()]
    return ", ".join(parts)


def _scenario_problem(s: ScenarioEvidence, known_tools: list[str], calls: list[CallRecord]) -> str:
    if s.tool not in known_tools:
        return (
            f"scenario {s.description!r} not exercised: no wrapped tool is named {s.tool!r} "
            f"(wrapped: {', '.join(known_tools) or 'none'})"
        )
    if not s.boundary_reached:
        called = sorted({c.tool for c in calls if not c.after_probe_end})
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
        self._signatures: dict[str, inspect.Signature | None] = {}
        self._known_tools: list[str] = []
        self._next_index = 0
        self._seq = 0
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
        """Stop injecting. Calls that start later pass straight through and are
        recorded in :attr:`late_calls`, never as evidence."""
        with self._lock:
            self._active = False
            self._closed = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def calls(self) -> list[CallRecord]:
        """Snapshot of calls that started while the probe was active, in
        invocation order. Pending calls are included."""
        with self._lock:
            return [dataclasses.replace(c) for c in self._calls]

    @property
    def late_calls(self) -> list[CallRecord]:
        with self._lock:
            return [dataclasses.replace(c) for c in self._late_calls]

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
        *,
        consumes_every_call: bool,
    ) -> None:
        with self._lock:
            if self._closed:
                raise ProbeError("cannot add scenarios to a closed probe")
            blocker = next(
                (r for r in self._rules if r.tool == tool and r.consumes_every_call), None
            )
            if blocker is not None:
                raise ProbeError(
                    f"scenario {description!r} can never run: the earlier scenario "
                    f"{blocker.description!r} applies to every call to {tool!r}. "
                    "Declare it after the other scenarios for this tool."
                )
            self._rules.append(
                _Rule(
                    len(self._rules) + 1,
                    tool,
                    description,
                    required,
                    planned,
                    decide,
                    consumes_every_call,
                )
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

    def _register(self, name: str, tool: Any = None) -> None:
        sig = _signature_of(tool) if tool is not None else None
        with self._lock:
            if name not in self._known_tools:
                self._known_tools.append(name)
                self._signatures[name] = sig
            elif self._signatures.get(name) != sig:
                self._signatures[name] = None  # two tools share a name: don't guess

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
            self._register(tool_name, tool)
            return self._wrap_callable(tool, tool_name)
        raise ProbeError(
            f"don't know how to wrap {type(tool).__name__}; pass a function or a LangChain tool"
        )

    def _wrap_callable(self, fn: Callable[..., Any], name: str) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await self._acall(name, fn, args, kwargs, key=None)

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._call(name, fn, args, kwargs, key=None)

        return wrapper

    def _wrap_langchain(self, tool: Any) -> Any:
        from langchain_core.tools import BaseTool, StructuredTool

        name = tool.name
        self._register(name, tool)
        clone = tool.model_copy()
        key = id(clone)  # identity of this wrapped tool, for delegation detection
        inner_run = clone._run

        @functools.wraps(inner_run)
        def run(*args: Any, **kwargs: Any) -> Any:
            if key in _ACTIVE_WRAPPERS.get():
                return inner_run(*args, **kwargs)  # same tool delegating to itself
            return self._call(name, inner_run, args, kwargs, key=key)

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
                if key in _ACTIVE_WRAPPERS.get():
                    return await inner_arun(*args, **kwargs)
                return await self._acall(name, inner_arun, args, kwargs, key=key)

            object.__setattr__(clone, "_arun", arun)
        return clone

    # ---- the call path -------------------------------------------------------------

    def _begin(self, name: str, args: tuple, kwargs: dict) -> tuple[CallRecord, _Action | None]:
        """Record the call at entry (pending) and decide its fault, atomically."""
        public_kwargs = _public_kwargs(kwargs)
        exact_args = tuple(_exact_or_opaque(v) for v in args)
        exact_kwargs = {k: _exact_or_opaque(v) for k, v in public_kwargs.items()}
        with self._lock:
            sig = self._signatures.get(name)
            self._seq += 1
            if not self._active:
                rec = CallRecord(
                    index=len(self._late_calls) + 1,
                    tool=name,
                    tool_call_index=0,
                    args=_jsonable(list(args)),
                    kwargs=_jsonable(_public_kwargs(kwargs)),
                    raw_args=tuple(args),
                    raw_kwargs=_public_kwargs(kwargs),
                    exact_args=exact_args,
                    exact_kwargs=exact_kwargs,
                    signature=sig,
                    started_seq=self._seq,
                    after_probe_end=True,
                )
                self._late_calls.append(rec)
                return rec, None
            self._next_index += 1
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
                index=self._next_index,
                tool=name,
                tool_call_index=n,
                args=_jsonable(list(args)),
                kwargs=_jsonable(_public_kwargs(kwargs)),
                raw_args=tuple(args),
                raw_kwargs=_public_kwargs(kwargs),
                exact_args=exact_args,
                exact_kwargs=exact_kwargs,
                signature=sig,
                injected=action is not None,
                fault=rule_hit.description if rule_hit else None,
                rule_id=rule_hit.rule_id if rule_hit else None,
                started_seq=self._seq,
            )
            self._calls.append(rec)
            return rec, action

    def _complete(
        self,
        rec: CallRecord,
        *,
        value: Any = None,
        exc: BaseException | None = None,
        started: float,
    ) -> None:
        with self._lock:
            self._seq += 1
            rec.status = "completed"
            rec.completed_seq = self._seq
            rec.duration_ms = (time.perf_counter() - started) * 1000
            if exc is None:
                rec.outcome = "returned"
                rec.value = _short(value)
            else:
                rec.outcome = "raised"
                rec.exception_type = type(exc).__name__
                rec.exception_message = str(exc)[:300]
            if not rec.after_probe_end and not self._active:
                rec.completed_after_close = True

    def _call(
        self, name: str, fn: Callable[..., Any], args: tuple, kwargs: dict, *, key: int | None
    ) -> Any:
        rec, action = self._begin(name, args, kwargs)
        started = time.perf_counter()
        token = _ACTIVE_WRAPPERS.set(_ACTIVE_WRAPPERS.get() | {key}) if key is not None else None
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
        except BaseException as exc:
            self._complete(rec, exc=exc, started=started)
            raise
        else:
            self._complete(rec, value=result, started=started)
            return result
        finally:
            if token is not None:
                _ACTIVE_WRAPPERS.reset(token)

    async def _acall(
        self, name: str, fn: Callable[..., Any], args: tuple, kwargs: dict, *, key: int | None
    ) -> Any:
        rec, action = self._begin(name, args, kwargs)
        started = time.perf_counter()
        token = _ACTIVE_WRAPPERS.set(_ACTIVE_WRAPPERS.get() | {key}) if key is not None else None
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
        except BaseException as exc:
            self._complete(rec, exc=exc, started=started)
            raise
        else:
            self._complete(rec, value=result, started=started)
            return result
        finally:
            if token is not None:
                _ACTIVE_WRAPPERS.reset(token)

    # ---- running the agent ---------------------------------------------------------

    def _check_runnable(self) -> None:
        if self._closed:
            raise ProbeError("this probe is closed; use a new probe for a new test")

    def run(
        self,
        agent: Any,
        input: Any = None,
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ProbeResult:
        """Run a **synchronous** agent on *input* and return a :class:`ProbeResult`.

        *agent* may be a function or an object with ``invoke`` (LangChain
        runnables, compiled LangGraph graphs, AgentProbe adapters). An async agent
        is also accepted when no event loop is running in this thread; inside a
        running loop (an async test) use ``await p.arun(...)`` instead, which this
        method tells you before doing anything.

        Exceptions from the agent are captured in the result, not raised. With
        ``timeout`` the agent runs in a worker thread. Python cannot stop a
        thread, so if the agent is still running at the deadline the result is
        marked ``deadline_exceeded`` and ``still_running`` and the probe closes.
        """
        self._check_runnable()
        call = _agent_callable(agent, prefer_async=False)
        if _loop_running() and _is_async_callable(call):
            raise ProbeError(
                "this agent is async and an event loop is already running here; "
                "use `await p.arun(agent, input)` instead of `p.run(...)`"
            )
        call_args = ((input,) if input is not None else ()) + args

        def target() -> Any:
            res = call(*call_args, **kwargs)
            if inspect.isawaitable(res):
                if _loop_running():
                    if inspect.iscoroutine(res):
                        res.close()  # never leave an un-awaited coroutine behind
                    raise ProbeError(
                        "the agent returned an awaitable inside a running event loop; "
                        "use `await p.arun(agent, input)`"
                    )
                return asyncio.run(_await(res))
            return res

        start = time.perf_counter()
        output: Any = None
        error: BaseException | None = None
        still_running = deadline_exceeded = False

        if timeout is None:
            try:
                output = target()
            except ProbeError:
                raise
            except Exception as exc:  # noqa: BLE001 - captured for assertions
                error = exc
        else:
            budget = _finite_number("timeout", timeout, low=0.0)
            ctx = contextvars.copy_context()
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = pool.submit(ctx.run, target)
            pool.shutdown(wait=False)
            done, _ = concurrent.futures.wait([future], timeout=budget)
            if done:
                # The agent finished in time. Whatever it raised (including its own
                # TimeoutError) is the agent's exception, not our deadline.
                try:
                    output = future.result()
                except ProbeError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    error = exc
            else:
                deadline_exceeded = True
                still_running = not future.done()
                if still_running:
                    self.close()
                else:  # finished in the tiny gap after the deadline
                    try:
                        output = future.result()
                    except Exception as exc:  # noqa: BLE001
                        error = exc
        return self._result(
            input, output, error, start, timeout, deadline_exceeded, still_running, cancelled=False
        )

    async def arun(
        self,
        agent: Any,
        input: Any = None,
        *args: Any,
        timeout: float | None = None,
        cancel_grace: float = 1.0,
        **kwargs: Any,
    ) -> ProbeResult:
        """Run an agent from async code and return a :class:`ProbeResult`.

        Invoking the agent and awaiting whatever it returns (an ``async def``
        agent, ``ainvoke``, or a sync callable that returns a coroutine) happen in
        **one task under one deadline**. Sync agents run in a worker thread.

        At the deadline the probe closes **first** (so no later tool call can be
        injected or counted), then cancellation is requested and given at most
        ``cancel_grace`` seconds. The result reports ``cancel_requested`` and
        ``cancelled`` (confirmed) separately. An agent that suppresses
        cancellation, or a sync agent's thread, is reported as ``still_running``.

        If the caller awaiting ``arun()`` is itself cancelled, the probe closes,
        the agent task is cancelled with the same bounded cleanup, and the
        caller's cancellation is re-raised.
        """
        self._check_runnable()
        call = _agent_callable(agent, prefer_async=True)
        call_args = ((input,) if input is not None else ()) + args
        budget = None if timeout is None else _finite_number("timeout", timeout, low=0.0)
        grace = _finite_number("cancel_grace", cancel_grace, low=0.0)
        thread_done = threading.Event()  # set when a sync agent's thread really finishes
        is_async = _is_async_callable(call)

        def run_sync() -> Any:
            try:
                return call(*call_args, **kwargs)
            finally:
                thread_done.set()

        async def invoke() -> Any:
            if is_async:
                result = call(*call_args, **kwargs)
                thread_done.set()  # no thread involved
            else:
                result = await asyncio.to_thread(run_sync)
            while inspect.isawaitable(result):  # returned coroutines stay under the deadline
                result = await result
            return result

        start = time.perf_counter()
        task = asyncio.ensure_future(invoke())
        output: Any = None
        error: BaseException | None = None
        deadline_exceeded = still_running = cancelled = cancel_requested = False

        try:
            done, _ = await asyncio.wait({task}, timeout=budget)
        except asyncio.CancelledError:
            # The caller was cancelled: stop injecting, stop the agent, propagate.
            self.close()
            task.cancel()
            await asyncio.wait({task}, timeout=grace)
            if task.done():
                _retrieve_quietly(task)
            else:
                task.add_done_callback(_retrieve_quietly)
            raise

        if done:
            try:
                output = task.result()
            except asyncio.CancelledError:
                cancelled = True
            except Exception as exc:  # noqa: BLE001 - captured for assertions
                error = exc
        else:
            deadline_exceeded = True
            self.close()  # before anything else: late tool calls can't be injected or counted
            task.cancel()
            cancel_requested = True
            finished, _ = await asyncio.wait({task}, timeout=grace)
            if finished:
                if task.cancelled():
                    cancelled = True  # confirmed
                else:
                    # It ignored cancellation but completed within the grace period.
                    try:
                        output = task.result()
                    except Exception as exc:  # noqa: BLE001
                        error = exc
            else:
                still_running = True  # suppressed cancellation and is still going
                task.add_done_callback(_retrieve_quietly)
            if not is_async and not thread_done.is_set():
                # Cancelling the task cannot stop a sync agent's thread.
                still_running = True
                cancelled = False
        return self._result(
            input,
            output,
            error,
            start,
            timeout,
            deadline_exceeded,
            still_running,
            cancelled=cancelled,
            cancel_requested=cancel_requested,
        )

    def _result(
        self,
        input: Any,
        output: Any,
        error: BaseException | None,
        start: float,
        timeout: float | None,
        deadline_exceeded: bool,
        still_running: bool,
        *,
        cancelled: bool,
        cancel_requested: bool = False,
    ) -> ProbeResult:
        elapsed_ms = (time.perf_counter() - start) * 1000
        finished = not (still_running or cancelled)
        answer = _answer_text(output) if (finished and error is None) else None
        result = ProbeResult(
            input=input,
            answer=answer,
            raw_output=output,
            exception=error,
            elapsed_ms=elapsed_ms,
            deadline_seconds=timeout,
            deadline_exceeded=deadline_exceeded,
            still_running=still_running,
            cancelled=cancelled,
            cancel_requested=cancel_requested,
            calls=self.calls,  # frozen snapshot, pending calls included
            scenarios=self.scenarios(),
            known_tools=list(self._known_tools),
        )
        with self._lock:
            self._results.append(result)
        return result

    # ---- gate ------------------------------------------------------------------------

    def gate(self) -> GateReport:
        """The gate over the whole probe: every required scenario exercised (from
        all calls that started while the probe was active) and every required
        assertion passed in every result."""
        with self._lock:
            assertions = [a for r in self._results for a in r.assertions]
        return _gate(self.scenarios(), assertions, list(self._known_tools), self.calls)


async def _await(aw: Any) -> Any:
    return await aw


def _retrieve_quietly(task: asyncio.Future[Any]) -> None:
    """Consume a straggling task's outcome so asyncio doesn't log it as unhandled."""
    if not task.cancelled():
        task.exception()


def _loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _is_async_callable(fn: Any) -> bool:
    if inspect.iscoroutinefunction(fn):
        return True
    dunder_call = getattr(fn, "__call__", None)  # noqa: B004 - objects with async __call__
    return dunder_call is not None and inspect.iscoroutinefunction(dunder_call)


def _agent_callable(agent: Any, *, prefer_async: bool) -> Callable[..., Any]:
    if prefer_async and hasattr(agent, "ainvoke"):
        return agent.ainvoke  # type: ignore[no-any-return]
    if hasattr(agent, "invoke"):
        return agent.invoke  # type: ignore[no-any-return]
    if hasattr(agent, "ainvoke"):
        return agent.ainvoke  # type: ignore[no-any-return]
    if callable(agent):
        return agent  # type: ignore[no-any-return]
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
