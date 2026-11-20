"""Regression tests: a probe never passes a test that did not test what it claims.

Each class covers one way a test could pass, or inject the wrong fault, silently.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import time

import pytest

from agentprobe.faults import ToolCallAPIError
from agentprobe.testing import ProbeError, probe


def lookup(order_id: str) -> str:
    return "shipped"


def agent_for(tools):
    return lambda q: tools["lookup"](q)


# ---------------------------------------------------------------------------
# A probe must be entered before it runs an agent
# ---------------------------------------------------------------------------


class TestMustEnter:
    def test_run_outside_with_raises(self):
        p = probe()
        tools = p.wrap({"lookup": lookup})
        p.on("lookup").always_fail()
        with pytest.raises(ProbeError, match="has not been entered"):
            p.run(agent_for(tools), "42")

    def test_arun_outside_with_raises(self):
        p = probe()

        async def agent(q):
            return "x"

        with pytest.raises(ProbeError, match="has not been entered"):
            asyncio.run(p.arun(agent, "42"))

    def test_closed_probe_still_says_closed(self):
        with probe() as p:
            pass
        with pytest.raises(ProbeError, match="closed"):
            p.run(lambda q: "x", "42")

    def test_inside_with_injects(self):
        with probe(enforce=False) as p:
            tools = p.wrap({"lookup": lookup})
            p.on("lookup").always_fail()
            r = p.run(agent_for(tools), "42")
        assert r.exception is not None


# ---------------------------------------------------------------------------
# with_=<instance> injects the exception exactly as configured
# ---------------------------------------------------------------------------


class GatewayError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


class NeedsTwo(Exception):
    def __new__(cls, code: int, message: str) -> NeedsTwo:
        return super().__new__(cls, message)

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)


def _injected(exc: BaseException, calls: int = 1) -> list[BaseException]:
    seen: list[BaseException] = []

    def agent(q):
        for _ in range(calls):
            try:
                tools["lookup"](q)
            except Exception as e:  # noqa: BLE001
                seen.append(e)
        return "done"

    with probe(enforce=False) as p:
        tools = p.wrap({"lookup": lookup})
        p.on("lookup").always_fail(with_=exc)
        p.run(agent, "42")
    return seen


class TestExceptionInstances:
    def test_builtin_fault_message_not_reformatted(self):
        (e,) = _injected(ToolCallAPIError(503))
        assert type(e) is ToolCallAPIError
        assert str(e) == "Tool API returned HTTP 503"
        assert e.status_code == 503

    def test_custom_init_keeps_type_and_attributes(self):
        (e,) = _injected(GatewayError(502, "bad gateway"))
        assert type(e) is GatewayError
        assert str(e) == "bad gateway"
        assert e.code == 502

    def test_os_errors_keep_errno(self):
        (e,) = _injected(ConnectionRefusedError(111, "refused"))
        assert type(e) is ConnectionRefusedError
        assert e.errno == 111 and e.strerror == "refused"
        (t,) = _injected(TimeoutError("timed out"))
        assert type(t) is TimeoutError and str(t) == "timed out"

    def test_each_injection_is_a_fresh_instance(self):
        original = GatewayError(500, "down")
        first, second = _injected(original, calls=2)
        assert first is not second and first is not original
        assert original.__traceback__ is None  # the configured instance is never raised

    def test_cause_and_notes_are_kept(self):
        exc = GatewayError(500, "down")
        exc.__cause__ = ValueError("root")
        exc.add_note("from the payment client")
        (e,) = _injected(exc)
        assert isinstance(e.__cause__, ValueError)
        assert e.__notes__ == ["from the payment client"]

    def test_uncopyable_instance_rejected_at_declaration(self):
        with probe(enforce=False) as p:
            with pytest.raises(ProbeError, match="cannot copy the exception"):
                p.on("lookup").always_fail(with_=NeedsTwo(1, "x"))


# ---------------------------------------------------------------------------
# Async callable objects are wrapped as async
# ---------------------------------------------------------------------------


class AsyncLookup:
    async def __call__(self, order_id: str) -> str:
        await asyncio.sleep(0)
        return "shipped"


class TestAsyncCallableObjects:
    def test_records_the_awaited_value(self):
        with probe(enforce=False) as p:
            tools = p.wrap({"lookup": AsyncLookup()})

            async def agent(q):
                return await tools["lookup"](q)

            r = p.run(agent, "42")
        assert r.answer == "shipped"
        (call,) = r.calls
        assert call.outcome == "returned"
        assert call.value == "'shipped'"  # the awaited value, not a coroutine

    def test_fault_injected_and_slow_does_not_block_the_loop(self):
        async def main():
            with probe(enforce=False) as p:
                tools = p.wrap({"lookup": AsyncLookup()})
                p.on("lookup").slow(ms=200, times=1).always_fail()
                ticks = 0

                async def ticker():
                    nonlocal ticks
                    for _ in range(10):
                        await asyncio.sleep(0.01)
                        ticks += 1

                async def agent(q):
                    t = asyncio.ensure_future(ticker())
                    first = await tools["lookup"](q)  # slowed
                    await t
                    try:
                        await tools["lookup"](q)
                    except Exception as e:  # noqa: BLE001
                        return f"{first} then {type(e).__name__}"
                    return "no fault"

                r = await p.arun(agent, "42")
                return r, ticks

        r, ticks = asyncio.run(main())
        assert r.answer == "shipped then ToolCallTimeout"
        assert ticks == 10  # the loop kept running during the 200 ms delay


# ---------------------------------------------------------------------------
# A blank answer is no answer
# ---------------------------------------------------------------------------


class TestBlankAnswer:
    @pytest.mark.parametrize("blank", ["", "   ", "\n"])
    def test_blank_never_passes_answer_assertions(self, blank):
        with probe(enforce=False) as p:
            r = p.run(lambda q: blank, "42")
            for check in (
                lambda: r.assert_answer_not_contains("refund"),
                lambda: r.assert_answer_not_matches("refund"),
                lambda: r.assert_answer(lambda a: True, "anything"),
            ):
                with pytest.raises(AssertionError, match="blank answer"):
                    check()
        assert r.answer == blank  # what the agent returned is still reported unchanged


# ---------------------------------------------------------------------------
# A hung agent never keeps the process alive
# ---------------------------------------------------------------------------


_HUNG = """
import asyncio, threading, time
from agentprobe.testing import probe
stop = threading.Event()
def hung(q):
    stop.wait(60)  # never set
    return "late"
with probe(enforce=False) as p:
    r = {call}
assert r.still_running, r
print("returned")
"""


@pytest.mark.parametrize(
    "call",
    [
        'p.run(hung, "x", timeout=0.1)',
        'asyncio.run(p.arun(hung, "x", timeout=0.1, cancel_grace=0.05))',
    ],
    ids=["run", "arun"],
)
def test_hung_sync_agent_does_not_block_exit(call):
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_HUNG).format(call=call)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    elapsed = time.perf_counter() - start
    assert proc.returncode == 0, proc.stderr
    assert "returned" in proc.stdout
    assert elapsed < 15, f"process took {elapsed:.1f}s to exit"
