"""Regression tests for the PR #98 review. Each test reproduces one reported case."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
import threading
import time
import warnings
from decimal import Decimal

import pytest

from agentprobe.injector.tool_failure import ToolCallTimeout
from agentprobe.testing import ProbeError, probe


def payment_status(order_id: str) -> str:
    return "confirmed"


# ---------------------------------------------------------------------------
# P1: a crashed agent must not pass answer assertions
# ---------------------------------------------------------------------------


class TestNoAnswerNeverPasses:
    def test_escaped_fault_fails_negative_answer_assertion(self):
        """Reported case: the tool is the agent, the fault escapes, answer is None."""
        with pytest.raises(AssertionError, match="no answer: the agent raised ToolCallTimeout"):
            with probe() as p:
                w = p.wrap(payment_status)
                p.on("payment_status").always_fail(with_="timeout")
                r = p.run(w, "42")
                r.assert_answer_not_contains("confirmed")
        assert r.answer is None and isinstance(r.exception, ToolCallTimeout)
        assert not p.gate().passed

    def test_agent_returning_none_fails_both_answer_assertions(self):
        with probe(enforce=False) as p:
            r = p.run(lambda q: None, "q")
            with pytest.raises(AssertionError, match="returned None"):
                r.assert_answer_not_contains("confirmed")
            with pytest.raises(AssertionError, match="returned None"):
                r.assert_answer_contains("confirmed")
        assert [a.status for a in r.assertions] == ["fail", "fail"]

    def test_still_running_is_unassessed_for_negative_assertion(self):
        release = threading.Event()
        with probe(enforce=False) as p:
            r = p.run(lambda q: release.wait(2) or "done", "q", timeout=0.03)
            with pytest.raises(AssertionError, match="unassessed"):
                r.assert_answer_not_contains("confirmed")
        release.set()
        assert r.assertions[0].status == "unassessed" and not p.gate().passed


# ---------------------------------------------------------------------------
# P1: in-flight calls are recorded at entry
# ---------------------------------------------------------------------------


class TestInFlightCalls:
    def test_call_in_flight_at_deadline_is_visible(self):
        """Reported case: tool starts, deadline expires, assert_not_called passed."""
        entered, release = threading.Event(), threading.Event()

        def blocking_tool(order_id: str) -> str:
            entered.set()
            release.wait(2)
            return "confirmed"

        with probe(enforce=False) as p:
            w = p.wrap(blocking_tool)
            p.on("blocking_tool").slow(ms=1)
            r = p.run(lambda q: w("42"), "q", timeout=0.2)
            assert entered.is_set() and r.still_running
            assert len(r.calls) == 1 and r.calls[0].status == "pending"
            with pytest.raises(AssertionError, match="1 still in flight"):
                r.assert_not_called("blocking_tool")
            r.assert_called("blocking_tool")
        assert r.scenarios[0].boundary_reached
        assert "IN FLIGHT" in r.summary()
        release.set()
        for _ in range(200):
            if p.calls[0].status == "completed":
                break
            time.sleep(0.01)
        # the probe's log is updated honestly; the earlier result snapshot is not
        assert p.calls[0].status == "completed" and p.calls[0].completed_after_close
        assert p.late_calls == []
        assert r.calls[0].status == "pending"

    def test_concurrent_calls_get_distinct_indexes(self):
        """Reported case: two simultaneous calls both received index=1."""
        barrier = threading.Barrier(2)

        def tool(x: int) -> int:
            barrier.wait(2)
            return x

        with probe(enforce=False) as p:
            w = p.wrap(tool)
            threads = [threading.Thread(target=w, args=(i,)) for i in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        calls = p.calls
        assert sorted(c.index for c in calls) == [1, 2]
        assert sorted(c.tool_call_index for c in calls) == [1, 2]
        assert len({c.started_seq for c in calls}) == 2

    def test_overlapping_calls_are_not_inferred_as_retries(self):
        """Retry/fallback inference pairs calls only when the failure completed
        before the next call started. Built deterministically from records."""
        from agentprobe.probe import CallRecord, ProbeResult

        def rec(i, tool, outcome, start, end):
            return CallRecord(
                index=i,
                tool=tool,
                tool_call_index=i,
                args=[],
                kwargs={},
                status="completed",
                outcome=outcome,
                started_seq=start,
                completed_seq=end,
            )

        def result(calls):
            return ProbeResult(
                input=None,
                answer="x",
                raw_output="x",
                exception=None,
                elapsed_ms=1.0,
                deadline_seconds=None,
                deadline_exceeded=False,
                still_running=False,
                calls=calls,
                scenarios=[],
                known_tools=[],
            )

        overlapping = [rec(1, "t", "raised", 1, 4), rec(2, "t", "returned", 2, 3)]
        sequential = [rec(1, "t", "raised", 1, 2), rec(2, "t", "returned", 3, 4)]
        fallback = [rec(1, "a", "raised", 1, 2), rec(2, "b", "returned", 3, 4)]
        assert result(overlapping).retries == {}
        assert result(sequential).retries == {"t": 1}
        assert result(fallback).fallbacks == [("a", "b")]


# ---------------------------------------------------------------------------
# P1: counts must be positive; unreachable rules rejected
# ---------------------------------------------------------------------------


class TestCountValidation:
    @pytest.mark.parametrize("bad", [0, -1, 1.5, math.inf, math.nan, True, "2"])
    def test_returns_times_rejected(self, bad):
        with pytest.raises(ProbeError, match="positive whole number"):
            probe().on("t").returns("x", times=bad)

    @pytest.mark.parametrize("bad", [0, -3, 2.5, math.nan, False])
    def test_slow_times_rejected(self, bad):
        with pytest.raises(ProbeError, match="positive whole number"):
            probe().on("t").slow(ms=10, times=bad)

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True])
    def test_fail_first_n_rejected(self, bad):
        with pytest.raises(ProbeError, match="positive whole number"):
            probe().on("t").fail_first(bad)

    @pytest.mark.parametrize("bad", [-1, math.inf, math.nan, "10"])
    def test_slow_ms_rejected(self, bad):
        with pytest.raises(ProbeError, match="finite number|must be >="):
            probe().on("t").slow(ms=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.1, math.nan, math.inf])
    def test_probability_rejected(self, bad):
        with pytest.raises(ProbeError):
            probe().on("t").with_probability(bad)

    def test_reported_zero_count_case_can_no_longer_pass(self):
        """Reported case: returns(..., times=0) was 'exercised' with zero faults."""
        with pytest.raises(ProbeError):
            with probe() as p:
                p.wrap(payment_status)
                p.on("payment_status").returns("stale", times=0)

    @pytest.mark.parametrize(
        "first",
        [
            lambda s: s.always_fail(),
            lambda s: s.returns("x", times=None),
            lambda s: s.slow(ms=1, times=None),
        ],
    )
    def test_rule_after_unbounded_rule_rejected(self, first):
        p = probe()
        first(p.on("t"))
        with pytest.raises(ProbeError, match="can never run"):
            p.on("t").fail_first(1)

    def test_unbounded_rule_last_is_fine(self):
        p = probe()
        p.on("t").fail_first(1).returns("stale", times=None)
        p.on("other").always_fail()  # a different tool is unaffected


# ---------------------------------------------------------------------------
# P2: nested LangChain tools
# ---------------------------------------------------------------------------


class TestNestedTools:
    def _tools(self):
        from langchain_core.tools import tool

        @tool
        def inner_tool(q: str) -> str:
            """Inner."""
            return "inner ok"

        return inner_tool

    def test_inner_wrapped_tool_still_gets_its_fault(self):
        """Reported case: an inner wrapped tool bypassed its always-fail rule."""
        from langchain_core.tools import tool

        inner = self._tools()
        with probe() as p:
            w_inner = p.wrap(inner)

            @tool
            def outer_tool(q: str) -> str:
                """Outer calls inner."""
                try:
                    return w_inner.invoke({"q": q})
                except ToolCallTimeout:
                    return "inner failed"

            w_outer = p.wrap(outer_tool)
            p.on("inner_tool").always_fail()
            assert w_outer.invoke({"q": "x"}) == "inner failed"
        assert [c.tool for c in p.calls] == ["outer_tool", "inner_tool"]
        assert p.scenarios()[0].injected == 1

    def test_nested_probes_on_nested_tools(self):
        from langchain_core.tools import tool

        inner = self._tools()
        with probe() as outer_p:
            with probe() as inner_p:
                w_inner = inner_p.wrap(inner)
                inner_p.on("inner_tool").fail_first(1)

                @tool
                def outer_tool(q: str) -> str:
                    """Outer."""
                    try:
                        return w_inner.invoke({"q": q})
                    except ToolCallTimeout:
                        return "inner failed"

                w_outer = outer_p.wrap(outer_tool)
                outer_p.on("outer_tool").returns("outer stubbed")
                assert w_outer.invoke({"q": "x"}) == "outer stubbed"
                assert w_outer.invoke({"q": "x"}) == "inner failed"

    def test_same_tool_async_to_sync_delegation_counted_once(self):
        from langchain_core.tools import tool

        @tool
        def lookup(q: str) -> str:
            """Sync only; ainvoke delegates to _run."""
            return "ok"

        with probe() as p:
            w = p.wrap(lookup)
            p.on("lookup").fail_first(1)

            async def go():
                with pytest.raises(ToolCallTimeout):
                    await w.ainvoke({"q": "1"})
                return await w.ainvoke({"q": "1"})

            assert asyncio.run(go()) == "ok"
        assert len(p.calls) == 2


# ---------------------------------------------------------------------------
# P2: upstream TimeoutError is not our deadline
# ---------------------------------------------------------------------------


class TestDeadlineAttribution:
    def test_immediate_upstream_timeout_is_not_deadline(self):
        """Reported case: TimeoutError with a 30s budget was labelled a deadline."""

        def agent(q):
            raise TimeoutError("upstream service timed out")

        with probe(enforce=False) as p:
            r = p.run(agent, "q", timeout=30)
        assert isinstance(r.exception, TimeoutError)
        assert not r.deadline_exceeded and not r.still_running
        assert r.elapsed_ms < 5000

    def test_real_deadline_is_deadline(self):
        release = threading.Event()
        with probe(enforce=False) as p:
            r = p.run(lambda q: release.wait(2), "q", timeout=0.05)
        release.set()
        assert r.deadline_exceeded and r.still_running and r.exception is None

    def test_timeout_must_be_valid(self):
        with probe(enforce=False) as p, pytest.raises(ProbeError):
            p.run(lambda q: "x", "q", timeout=math.nan)


# ---------------------------------------------------------------------------
# P2: native async
# ---------------------------------------------------------------------------


class TestNativeAsync:
    async def test_arun_in_running_loop(self):
        async def fetch(order_id: str) -> str:
            await asyncio.sleep(0)
            return "confirmed"

        async with probe() as p:
            t = p.wrap(fetch)
            p.on("fetch").fail_first(1)

            async def agent(q):
                for _ in range(2):
                    try:
                        return f"Order {q} is {await t(q)}"
                    except ToolCallTimeout:
                        continue
                return "gave up"

            r = await p.arun(agent, "42")
            r.assert_no_crash().assert_answer_contains("confirmed")
        assert r.retries == {"fetch": 1}

    async def test_sync_run_inside_loop_is_rejected_without_warnings(self):
        async def agent(q):
            return "x"

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # an un-awaited coroutine warning would fail here
            async with probe(enforce=False) as p:
                with pytest.raises(ProbeError, match="await p.arun"):
                    p.run(agent, "q")

    async def test_arun_cancels_async_agent_at_deadline(self):
        cancelled = asyncio.Event()

        async def slow_agent(q):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async with probe(enforce=False) as p:
            r = await p.arun(slow_agent, "q", timeout=0.05)
        assert r.deadline_exceeded and r.cancelled and not r.still_running
        assert cancelled.is_set()
        with pytest.raises(AssertionError, match="unassessed"):
            r.assert_answer_contains("x")

    async def test_arun_with_sync_agent(self):
        async with probe(enforce=False) as p:
            r = await p.arun(lambda q: f"echo {q}", "hi")
        assert r.answer == "echo hi"

    async def test_arun_with_ainvoke_object(self):
        class Agent:
            async def ainvoke(self, x):
                return {"output": f"async {x}"}

        async with probe(enforce=False) as p:
            r = await p.arun(Agent(), "hi")
        assert r.answer == "async hi"

    async def test_arun_captures_agent_exception(self):
        async def agent(q):
            raise ValueError("boom")

        async with probe(enforce=False) as p:
            r = await p.arun(agent, "q")
        assert isinstance(r.exception, ValueError) and not r.deadline_exceeded


# ---------------------------------------------------------------------------
# P2: JSON-safe results
# ---------------------------------------------------------------------------


class TestJsonSafety:
    def test_non_json_arguments_serialize(self):
        """Reported case: a datetime argument made json.dumps(result.to_dict()) fail."""

        class Custom:
            def __repr__(self) -> str:
                return "Custom()"

        def tool(when, blob, obj, amount):
            return {"at": when, "n": len(blob)}

        when = dt.datetime(2026, 11, 17, 12, 0)
        blob = b"\x00\x01payload"
        obj = Custom()
        with probe(enforce=False) as p:
            w = p.wrap(tool)
            r = p.run(lambda q: w(when, blob, obj, Decimal("1.50")) and "done", "q")
        text = json.dumps(r.to_dict())
        d = json.loads(text)
        args = d["observed"]["calls"][0]["args"]
        assert args[0] == "2026-11-17T12:00:00"
        assert args[1] == {"type": "bytes", "length": 9, "preview": blob.hex()}
        assert args[2]["type"].endswith("Custom") and args[2]["repr"] == "Custom()"
        assert args[3] == "1.50"
        # caller objects untouched; the live references are kept separately
        assert isinstance(r.calls[0].raw_args[0], dt.datetime) and r.calls[0].raw_args[2] is obj

    def test_unrepresentable_object_does_not_break_reporting(self):
        class Bad:
            def __repr__(self):
                raise RuntimeError("no repr")

        with probe(enforce=False) as p:
            w = p.wrap({"tool": lambda x: "ok"})["tool"]
            r = p.run(lambda q: w(Bad()), "q")
        d = json.loads(json.dumps(r.to_dict()))
        assert "unrepresentable Bad" in d["observed"]["calls"][0]["args"][0]["repr"]
        assert "unrepresentable Bad" in r.summary()


# ---------------------------------------------------------------------------
# Re-review: async lifetime under one deadline, bounded cancellation, caller
# cancellation, frozen argument evidence
# ---------------------------------------------------------------------------


class TestAsyncLifetime:
    async def test_sync_callable_returning_coroutine_is_under_the_deadline(self):
        """Reported case: a 40 ms coroutine returned by a sync lambda, timeout 5 ms,
        came back after ~41 ms with deadline_exceeded=False."""

        async def slow():
            await asyncio.sleep(0.04)
            return "late answer"

        async with probe(enforce=False) as p:
            r = await p.arun(lambda q: slow(), "q", timeout=0.005)
        assert r.deadline_exceeded and r.cancel_requested and r.cancelled
        assert r.answer is None and r.elapsed_ms < 1000

    async def test_returned_awaitable_object_is_awaited(self):
        loop = asyncio.get_running_loop()

        def agent(q):
            fut = loop.create_future()
            loop.call_later(0.01, fut.set_result, f"future {q}")
            return fut

        async with probe(enforce=False) as p:
            r = await p.arun(agent, "q", timeout=2)
        assert r.answer == "future q" and not r.deadline_exceeded

    async def test_agent_suppressing_cancellation_is_bounded_and_isolated(self):
        """Reported case: an agent caught CancelledError, arun waited forever, and a
        tool call after the deadline was injected and counted as evidence."""
        release = asyncio.Event()

        async def tool(x: str) -> str:
            return "ok"

        async with probe(enforce=False) as p:
            t = p.wrap({"tool": tool})["tool"]
            p.on("tool").always_fail()

            async def stubborn(q):
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    await release.wait()  # ignores cancellation, keeps going
                    return await t("after deadline")

            started = time.perf_counter()
            r = await p.arun(stubborn, "q", timeout=0.02, cancel_grace=0.05)
            assert time.perf_counter() - started < 2  # bounded
            assert not p.active  # closed before cleanup
        assert r.deadline_exceeded and r.cancel_requested
        assert r.still_running and not r.cancelled
        release.set()
        for _ in range(100):
            if p.late_calls:
                break
            await asyncio.sleep(0.01)
        late = p.late_calls
        assert late and late[0].after_probe_end and not late[0].injected
        assert p.scenarios()[0].injected == 0  # never counted as evidence
        assert p.calls == []

    async def test_agent_that_finishes_during_grace_is_reported_honestly(self):
        async def ignores_then_finishes(q):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                return "finished anyway"

        async with probe(enforce=False) as p:
            r = await p.arun(ignores_then_finishes, "q", timeout=0.02, cancel_grace=1)
        assert r.deadline_exceeded and r.cancel_requested
        assert not r.cancelled and not r.still_running and r.answer == "finished anyway"
        with pytest.raises(AssertionError, match="finished within"):
            r.assert_within(0.02)

    async def test_caller_cancellation_cancels_async_agent(self):
        """Reported case: cancelling the caller left the agent task running."""
        agent_cancelled = asyncio.Event()
        agent_started = asyncio.Event()

        async def agent(q):
            agent_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                agent_cancelled.set()
                raise

        p = probe(enforce=False)
        async with p:
            caller = asyncio.ensure_future(p.arun(agent, "q", cancel_grace=1))
            await agent_started.wait()
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert agent_cancelled.is_set()
            assert not p.active

    async def test_caller_cancellation_with_sync_agent_in_thread(self):
        release = threading.Event()
        entered = threading.Event()

        def sync_agent(q):
            entered.set()
            release.wait(2)
            return "done"

        p = probe(enforce=False)
        async with p:
            caller = asyncio.ensure_future(p.arun(sync_agent, "q", cancel_grace=0.05))
            while not entered.is_set():
                await asyncio.sleep(0.005)
            caller.cancel()
            started = time.perf_counter()
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert time.perf_counter() - started < 2  # bounded, the thread can't be stopped
            assert not p.active
        release.set()

    async def test_sync_agent_past_deadline_is_still_running(self):
        release = threading.Event()
        async with probe(enforce=False) as p:
            r = await p.arun(lambda q: release.wait(2) or "x", "q", timeout=0.02, cancel_grace=0.02)
        release.set()
        assert r.deadline_exceeded and r.still_running and not r.cancelled

    async def test_invalid_cancel_grace_rejected(self):
        async with probe(enforce=False) as p:
            with pytest.raises(ProbeError):
                await p.arun(lambda q: "x", "q", cancel_grace=-1)


class TestFrozenArguments:
    def test_mutating_arguments_after_the_run_does_not_change_evidence(self):
        """Reported case: mutating an input dict after the run changed the earlier
        result's serialized evidence."""
        payload = {"status": "initial", "items": [1, 2]}
        with probe(enforce=False) as p:
            w = p.wrap({"tool": lambda d: "ok"})["tool"]
            r = p.run(lambda q: w(payload), "q")
        before = json.dumps(r.to_dict(), sort_keys=True)
        summary_before = r.summary()
        payload["status"] = "changed later"
        payload["items"].append(3)
        assert json.dumps(r.to_dict(), sort_keys=True) == before
        assert r.summary() == summary_before
        assert r.calls[0].args == [{"status": "initial", "items": [1, 2]}]
        assert p.calls[0].args == [{"status": "initial", "items": [1, 2]}]
        # the live reference is still available, separately, and does change
        assert r.calls[0].raw_args[0]["status"] == "changed later"

    def test_uncopyable_arguments_are_fine(self):
        lock = threading.Lock()
        with probe(enforce=False) as p:
            w = p.wrap({"tool": lambda resource: "ok"})["tool"]
            r = p.run(lambda q: w(lock), "q")
        assert r.calls[0].raw_args[0] is lock
        assert "lock" in r.calls[0].args[0]["repr"].lower()
        json.dumps(r.to_dict())
