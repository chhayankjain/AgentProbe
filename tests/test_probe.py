"""Tests for agentprobe.probe: wrap_tools (#63), scenarios (#65), ProbeResult (#66),
scenario evidence (#94), instrumentation contract (#95), minimal pytest workflow (#96)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

import pytest

from agentprobe.injector.tool_failure import (
    ToolCallAPIError,
    ToolCallError,
    ToolCallRateLimited,
    ToolCallTimeout,
)
from agentprobe.testing import Probe, ProbeError, ProbeGateError, probe

# ---------------------------------------------------------------------------
# Helpers: tools and agents with no model
# ---------------------------------------------------------------------------


def payment_status(order_id: str) -> str:
    return "confirmed"


def send_receipt(order_id: str) -> str:
    return "sent"


def careful_agent(tools: dict) -> Any:
    """Retries once, then admits failure."""

    def run(question: str) -> str:
        for _ in range(2):
            try:
                return f"Order 42 payment is {tools['payment_status']('42')}."
            except Exception:
                continue
        return "I could not confirm the payment for order 42."

    return run


def broken_agent(tools: dict) -> Any:
    """Invents success when the tool fails."""

    def run(question: str) -> str:
        try:
            status = tools["payment_status"]("42")
        except Exception:
            status = "confirmed"
        return f"Order 42 payment is {status}."

    return run


# ---------------------------------------------------------------------------
# Scenarios and evidence (#65, #94)
# ---------------------------------------------------------------------------


class TestScenarios:
    def test_agent_can_retry_a_named_timeout_with_its_standard_handler(self):
        with probe() as p:
            check = p.wrap(payment_status)
            p.on(check).fail_first(1, with_="timeout")

            def agent(order_id):
                try:
                    return check(order_id)
                except TimeoutError as exc:
                    assert isinstance(exc, ToolCallError)
                    return check(order_id)

            result = p.run(agent, "42")
            result.assert_answer_contains("confirmed")
        first, retry = result.calls
        assert first.injected and first.exception_type == "ToolCallTimeout"
        assert retry.outcome == "returned" and not retry.injected

    def test_fail_first_then_recover(self):
        with probe() as p:
            t = p.wrap({"payment_status": payment_status})
            p.on("payment_status").fail_first(1, with_="timeout")
            r = p.run(careful_agent(t), "Did order 42 go through?")
            r.assert_answer_contains("confirmed")
        assert [c.outcome for c in r.calls] == ["raised", "returned"]
        assert r.calls[0].exception_type == "ToolCallTimeout" and r.calls[0].injected
        assert r.scenarios[0].status == "exercised"
        assert r.retries == {"payment_status": 1}

    def test_fail_times_then_succeed_is_fail_first(self):
        with probe() as p:
            t = p.wrap({"payment_status": payment_status})
            p.on("payment_status").fail_times(2).then_succeed()
            calls = []
            for _ in range(3):
                try:
                    calls.append(t["payment_status"]("1"))
                except ToolCallTimeout:
                    calls.append("timeout")
        assert calls == ["timeout", "timeout", "confirmed"]

    @pytest.mark.parametrize(
        "name,exc",
        [
            ("timeout", ToolCallTimeout),
            ("rate_limit", ToolCallRateLimited),
            ("api_error", ToolCallAPIError),
        ],
    )
    def test_named_faults(self, name, exc):
        with probe() as p:
            t = p.wrap({"payment_status": payment_status})
            p.on("payment_status").always_fail(with_=name)
            with pytest.raises(exc):
                t["payment_status"]("1")

    def test_custom_exception_class_and_instance(self):
        class Boom(Exception):
            pass

        with probe() as p:
            t = p.wrap({"a": payment_status, "b": send_receipt})
            p.on("a").fail_first(1, with_=Boom)
            p.on("b").fail_first(1, with_=ValueError("bad payload"))
            with pytest.raises(Boom):
                t["a"]("1")
            with pytest.raises(ValueError, match="bad payload"):
                t["b"]("1")

    def test_returns_value_instead_of_calling(self):
        called = []

        def real(order_id: str) -> dict:
            called.append(order_id)
            return {"status": "confirmed"}

        with probe() as p:
            t = p.wrap({"payment_status": real})
            p.on("payment_status").returns({"status": None})
            assert t["payment_status"]("1") == {"status": None}
            assert t["payment_status"]("2") == {"status": "confirmed"}
        assert called == ["2"]

    def test_slow_delays_then_runs(self):
        with probe() as p:
            t = p.wrap({"payment_status": payment_status})
            p.on("payment_status").slow(ms=50)
            start = time.perf_counter()
            assert t["payment_status"]("1") == "confirmed"
            assert time.perf_counter() - start >= 0.045
        assert p.calls[0].injected

    def test_probability_is_seeded(self):
        def outcomes(seed: int) -> list[bool]:
            with probe(enforce=False) as p:
                t = p.wrap({"payment_status": payment_status})
                p.on("payment_status").with_probability(0.5, seed=seed)
                for _ in range(20):
                    try:
                        t["payment_status"]("1")
                    except ToolCallTimeout:
                        pass
            return [c.injected for c in p.calls]

        assert outcomes(7) == outcomes(7)
        assert any(outcomes(7)) and not all(outcomes(7))

    def test_rules_for_same_tool_form_a_sequence(self):
        with probe() as p:
            t = p.wrap({"payment_status": payment_status})
            p.on("payment_status").fail_first(1, with_="timeout").returns("stale", times=2)
            with pytest.raises(ToolCallTimeout):
                t["payment_status"]("1")
            assert [t["payment_status"]("1") for _ in range(3)] == ["stale", "stale", "confirmed"]
        assert [s.status for s in p.scenarios()] == ["exercised", "exercised"]

    def test_invalid_scenarios_rejected(self):
        p = Probe()
        with pytest.raises(ProbeError):
            p.on("x").fail_first(0)
        with pytest.raises(ProbeError):
            p.on("x").fail_first(1, with_="nope")
        with pytest.raises(ProbeError):
            p.on("x").with_probability(1.5)


class TestEvidence:
    def test_bypassing_agent_fails_the_gate(self):
        with pytest.raises(ProbeGateError, match="never called 'payment_status'"):
            with probe() as p:
                p.wrap({"payment_status": payment_status})
                p.on("payment_status").fail_first(1)
                r = p.run(lambda q: "Cached: order 42 confirmed.", "q")
                r.assert_answer_contains("confirmed")  # passes, but the fault never fired

    def test_cached_answer_after_first_call_is_partial(self):
        with pytest.raises(ProbeGateError, match="partially exercised"):
            with probe() as p:
                t = p.wrap({"payment_status": payment_status})
                p.on("payment_status").fail_first(2)
                try:
                    t["payment_status"]("1")
                except ToolCallTimeout:
                    pass  # agent gives up after one try: only 1 of 2 planned faults fired
        assert p.scenarios()[0].status == "partially_exercised"

    def test_unknown_tool_name_explained(self):
        with pytest.raises(ProbeGateError, match="no wrapped tool is named 'payments'"):
            with probe() as p:
                p.wrap({"payment_status": payment_status})
                p.on("payments").fail_first(1)

    def test_optional_scenario_does_not_gate(self):
        with probe() as p:
            p.wrap({"payment_status": payment_status})
            p.on("payment_status").fail_first(1, required=False)
        assert p.scenarios()[0].status == "not_exercised"
        assert p.gate().passed

    def test_probability_rule_with_no_fault_is_not_exercised(self):
        with pytest.raises(ProbeGateError):
            with probe() as p:
                t = p.wrap({"payment_status": payment_status})
                p.on("payment_status").with_probability(0.0, seed=1)
                t["payment_status"]("1")

    def test_evidence_fields(self):
        with probe() as p:
            t = p.wrap({"payment_status": payment_status})
            p.on("payment_status").fail_first(1)
            p.run(careful_agent(t), "q")
        ev = p.scenarios()[0]
        assert (ev.boundary_reached, ev.planned, ev.injected, ev.tool_calls, ev.required) == (
            True,
            1,
            1,
            2,
            True,
        )


# ---------------------------------------------------------------------------
# Gate rules (#74 semantics, applied at probe exit)
# ---------------------------------------------------------------------------


class TestGate:
    def test_broken_agent_is_caught(self):
        with pytest.raises(AssertionError, match="does not contain 'confirmed'"):
            with probe() as p:
                t = p.wrap({"payment_status": payment_status})
                p.on("payment_status").always_fail(with_="timeout")
                r = p.run(broken_agent(t), "Did order 42 go through?")
                r.assert_answer_not_contains("confirmed")

    def test_fixed_agent_passes_same_test(self):
        with probe() as p:
            t = p.wrap({"payment_status": payment_status})
            p.on("payment_status").always_fail(with_="timeout")
            r = p.run(careful_agent(t), "Did order 42 go through?")
            r.assert_answer_not_contains("confirmed")
            r.assert_no_crash()
        assert p.gate().passed

    def test_unassessed_required_assertion_fails_gate(self):
        def slow(q):
            time.sleep(0.4)
            return "done"

        p = probe(enforce=False)
        with p:
            r = p.run(slow, "q", timeout=0.05)
            with pytest.raises(AssertionError, match="unassessed"):
                r.assert_answer_contains("done")
        report = p.gate()
        assert not report.passed and any("unassessed" in f for f in report.failures)

    def test_swallowed_assertion_still_fails_gate(self):
        with pytest.raises(ProbeGateError, match="is fail"):
            with probe() as p:
                t = p.wrap({"payment_status": payment_status})
                p.on("payment_status").fail_first(1)
                r = p.run(careful_agent(t), "q")
                try:
                    r.assert_answer_contains("refunded")
                except AssertionError:
                    pass

    def test_non_required_assertion_never_gates(self):
        with probe() as p:
            t = p.wrap({"payment_status": payment_status})
            p.on("payment_status").fail_first(1)
            r = p.run(careful_agent(t), "q")
            r.assert_answer_contains("refunded", required=False)
        assert p.gate().passed

    def test_exception_in_block_is_not_masked_by_gate(self):
        with pytest.raises(KeyError):
            with probe() as p:
                p.wrap({"payment_status": payment_status})
                p.on("payment_status").fail_first(1)
                raise KeyError("user bug")


# ---------------------------------------------------------------------------
# ProbeResult (#66)
# ---------------------------------------------------------------------------


class TestProbeResult:
    def test_agent_exception_captured_not_raised(self):
        def crashing(q):
            raise RuntimeError("agent bug")

        with probe(enforce=False) as p:
            r = p.run(crashing, "q")
        assert isinstance(r.exception, RuntimeError) and r.answer is None
        with pytest.raises(AssertionError, match="RuntimeError"):
            r.assert_no_crash()

    def test_fallback_inferred(self):
        def agent(tools):
            def run(q):
                try:
                    return tools["primary"]("1")
                except Exception:
                    return tools["backup"]("1")

            return run

        with probe() as p:
            t = p.wrap({"primary": payment_status, "backup": send_receipt})
            p.on("primary").always_fail()
            r = p.run(agent(t), "q")
        assert r.fallbacks == [("primary", "backup")] and r.retries == {}

    def test_answer_extraction(self):
        from langchain_core.messages import AIMessage

        with probe(enforce=False) as p:
            assert p.run(lambda q: {"output": "x"}, "q").answer == "x"
            assert p.run(lambda q: {"messages": [AIMessage(content="m")]}, "q").answer == "m"
            assert (
                p.run(lambda q: AIMessage(content=[{"type": "text", "text": "blk"}]), "q").answer
                == "blk"
            )

    def test_runnable_with_invoke(self):
        class Agent:
            def invoke(self, x):
                return f"echo {x}"

        with probe(enforce=False) as p:
            assert p.run(Agent(), "hi").answer == "echo hi"

    def test_async_agent_and_async_tool(self):
        async def fetch(order_id: str) -> str:
            await asyncio.sleep(0)
            return "confirmed"

        with probe() as p:
            t = p.wrap({"fetch": fetch})
            p.on("fetch").fail_first(1)

            async def agent(q):
                for _ in range(2):
                    try:
                        return await t["fetch"]("1")
                    except ToolCallTimeout:
                        continue
                return "gave up"

            r = p.run(agent, "q")
            r.assert_answer_contains("confirmed")

    def test_summary_has_three_sections_and_json_is_versioned(self):
        with probe() as p:
            t = p.wrap({"payment_status": payment_status})
            p.on("payment_status").fail_first(1)
            r = p.run(careful_agent(t), "q")
            r.assert_answer_contains("confirmed")
        s = r.summary()
        assert s.index("Observed") < s.index("Asserted") < s.index("Inferred")
        d = json.loads(json.dumps(r.to_dict(), default=str))
        assert d["schema_version"] == "1.2"
        assert set(d) >= {"observed", "asserted", "inferred", "gate"}
        assert d["gate"]["passed"] is True

    def test_framework_plumbing_not_logged(self):
        with probe(enforce=False) as p:
            t = p.wrap({"f": lambda **kw: "ok"})
            t["f"](x=1, run_manager="rm", callbacks="cb", config="c")
        assert p.calls[0].kwargs == {"x": 1}


# ---------------------------------------------------------------------------
# Instrumentation contract (#95)
# ---------------------------------------------------------------------------


def _lc_tool():
    from langchain_core.tools import tool

    @tool
    def payment_status_tool(order_id: str) -> str:
        """Return the payment status for an order."""
        return "confirmed"

    return payment_status_tool


class TestInstrumentation:
    def test_plain_function_untouched(self):
        before = dict(vars(payment_status))
        with probe(enforce=False) as p:
            w = p.wrap(payment_status)
            assert w is not payment_status and w.__wrapped__ is payment_status
        assert dict(vars(payment_status)) == before

    def test_langchain_tool_restoration_and_schema(self):
        from langchain_core.utils.function_calling import convert_to_openai_tool

        t = _lc_tool()
        run_before, arun_before = t._run, t._arun
        vars_before = {k: v for k, v in vars(t).items()}
        with probe() as p:
            w = p.wrap([t])[0]
            assert w is not t
            assert convert_to_openai_tool(w) == convert_to_openai_tool(t)
            p.on(t).fail_first(1)
            with pytest.raises(ToolCallTimeout):
                w.invoke({"order_id": "42"})
            assert w.invoke({"order_id": "42"}) == "confirmed"
        assert t._run == run_before and t._arun == arun_before
        assert {k: v for k, v in vars(t).items()} == vars_before
        assert "_run" not in vars(t) and "_arun" not in vars(t)
        assert not any(isinstance(v, Probe) for v in vars(t).values())

    def test_langchain_async_tool_counted_once(self):
        from langchain_core.tools import tool

        @tool
        async def lookup(order_id: str) -> str:
            """Look up an order."""
            return "confirmed"

        with probe() as p:
            w = p.wrap(lookup)
            p.on("lookup").fail_first(1)

            async def go():
                with pytest.raises(ToolCallTimeout):
                    await w.ainvoke({"order_id": "1"})
                return await w.ainvoke({"order_id": "1"})

            assert asyncio.run(go()) == "confirmed"
        assert len(p.calls) == 2

    def test_sync_tool_via_ainvoke_counted_once(self):
        t = _lc_tool()
        with probe() as p:
            w = p.wrap(t)
            p.on(t).fail_first(1)

            async def go():
                with pytest.raises(ToolCallTimeout):
                    await w.ainvoke({"order_id": "1"})

            asyncio.run(go())
        assert len(p.calls) == 1

    def test_cleanup_after_exception(self):
        with pytest.raises(RuntimeError):
            with probe() as p:
                w = p.wrap(payment_status)
                p.on("payment_status").always_fail()
                raise RuntimeError("test bug")
        assert not p.active
        assert w("1") == "confirmed"  # closed probe: pass-through
        assert p.late_calls[0].after_probe_end and not p.late_calls[0].injected

    def test_cleanup_after_cancellation(self):
        holder: dict = {}

        async def go():
            async with probe() as p:
                holder["probe"] = p
                holder["tool"] = p.wrap(payment_status)
                p.on("payment_status").always_fail()
                raise asyncio.CancelledError

        async def outer():
            try:
                await go()
            except asyncio.CancelledError:
                holder["cancelled"] = True

        asyncio.run(outer())
        p = holder["probe"]
        assert holder["cancelled"] and not p.active  # closed, and the gate did not mask it
        assert holder["tool"]("1") == "confirmed"  # pass-through after close
        assert p.late_calls[0].after_probe_end and not p.late_calls[0].injected

    def test_still_running_agent_late_calls_never_count(self):
        release = threading.Event()

        with probe(enforce=False) as p:
            t = p.wrap({"payment_status": payment_status})
            p.on("payment_status").fail_first(1)

            def slow_agent(q):
                release.wait(2)
                return t["payment_status"]("1")

            r = p.run(slow_agent, "q", timeout=0.05)
        assert r.still_running and r.deadline_exceeded and r.answer is None
        release.set()
        for _ in range(100):
            if p.late_calls:
                break
            time.sleep(0.01)
        late = p.late_calls
        assert late and late[0].after_probe_end and not late[0].injected
        assert p.scenarios()[0].status == "not_exercised"
        assert p.calls == []
        with pytest.raises(ProbeError):
            p.run(lambda q: "x", "q")

    def test_late_call_does_not_leak_into_a_new_probe(self):
        release = threading.Event()
        with probe(enforce=False) as p1:
            t1 = p1.wrap({"payment_status": payment_status})

            def slow_agent(q):
                release.wait(2)
                return t1["payment_status"]("1")

            p1.run(slow_agent, "q", timeout=0.05)
        with probe() as p2:
            t2 = p2.wrap({"payment_status": payment_status})
            p2.on("payment_status").fail_first(1)
            release.set()
            time.sleep(0.05)
            with pytest.raises(ToolCallTimeout):
                t2["payment_status"]("1")
        assert len(p2.calls) == 1 and p2.scenarios()[0].injected == 1

    def test_concurrent_probes_on_same_original_are_isolated(self):
        results: dict[str, list[str]] = {}
        barrier = threading.Barrier(2)

        def worker(name: str, fail: bool):
            with probe(enforce=False) as p:
                w = p.wrap(payment_status)
                if fail:
                    p.on("payment_status").always_fail()
                barrier.wait()
                out = []
                for _ in range(5):
                    try:
                        out.append(w("1"))
                    except ToolCallTimeout:
                        out.append("timeout")
                results[name] = out

        a = threading.Thread(target=worker, args=("a", True))
        b = threading.Thread(target=worker, args=("b", False))
        a.start()
        b.start()
        a.join()
        b.join()
        assert results["a"] == ["timeout"] * 5
        assert results["b"] == ["confirmed"] * 5

    def test_nested_probes_wrap_independently(self):
        with probe() as outer:
            w1 = outer.wrap(payment_status)
            outer.on("payment_status").fail_first(1)
            with probe() as inner:
                w2 = inner.wrap(payment_status)
                inner.on("payment_status").returns("stale")
                assert w2("1") == "stale"
            with pytest.raises(ToolCallTimeout):
                w1("1")

    def test_probe_cannot_be_reused(self):
        p = probe(enforce=False)
        with p:
            pass
        with pytest.raises(ProbeError):
            with p:
                pass

    def test_wrap_containers_and_errors(self):
        with probe(enforce=False) as p:
            assert isinstance(p.wrap([payment_status]), list)
            assert isinstance(p.wrap((payment_status,)), tuple)
            assert set(p.wrap({"x": payment_status})) == {"x"}
            with pytest.raises(ProbeError):
                p.wrap(42)


# ---------------------------------------------------------------------------
# Real framework: LangChain tools inside a LangGraph agent, unchanged
# ---------------------------------------------------------------------------


def _react_agent(tools, handle_errors: bool):
    import warnings

    from agentprobe.fake_llm import ScriptedToolCallingModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langgraph.prebuilt import ToolNode, create_react_agent

    model = ScriptedToolCallingModel(tool_name="payment_status_tool", tool_args={"order_id": "42"})
    node = ToolNode(tools, handle_tool_errors=True) if handle_errors else tools
    return create_react_agent(model, node)  # the user's agent, built as usual


QUESTION_MSGS = {"messages": [("user", "Did order 42 go through?")]}


def test_default_langgraph_agent_crashes_on_tool_timeout():
    """A real finding the probe surfaces: LangGraph's prebuilt agent does not catch
    arbitrary tool exceptions by default, so one tool timeout crashes the agent."""
    t = _lc_tool()
    with pytest.raises(AssertionError, match="no unhandled exception: ToolCallTimeout"):
        with probe() as p:
            tools = p.wrap([t])
            p.on("payment_status_tool").always_fail(with_="timeout")
            r = p.run(_react_agent(tools, handle_errors=False), QUESTION_MSGS)
            r.assert_no_crash()
    assert r.calls[0].tool == "payment_status_tool" and r.calls[0].injected
    assert r.scenarios[0].status == "exercised"


def test_langgraph_agent_with_tool_error_handling_admits_failure():
    t = _lc_tool()
    with probe() as p:
        tools = p.wrap([t])
        p.on("payment_status_tool").always_fail(with_="timeout")
        r = p.run(_react_agent(tools, handle_errors=True), QUESTION_MSGS)
        r.assert_no_crash()
        r.assert_answer_contains("could not complete")


def test_no_scenarios_means_identical_behaviour():
    """#63: a wrapped tool with no faults declared behaves exactly like the original,
    including raising the original's own exceptions."""

    def divide(a: float, b: float) -> float:
        return a / b

    with probe() as p:
        w = p.wrap(divide)
        assert w(6, 3) == divide(6, 3)
        with pytest.raises(ZeroDivisionError):
            w(1, 0)
    assert [c.injected for c in p.calls] == [False, False]
    assert p.calls[1].exception_type == "ZeroDivisionError"
