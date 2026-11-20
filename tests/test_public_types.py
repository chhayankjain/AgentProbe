"""Static types of the public injector API match what it returns at runtime.

CI type-checks this file with ``mypy --strict`` (``assert_type`` fails the check
if an inferred type is wrong) and pytest runs it, so a signature that lies about
its return type is caught either way.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, assert_type

from agentprobe.injector.latency_failure import (
    LatencyFailureConfig,
    LatencyFailureInjector,
    LatencyFailureType,
)
from agentprobe.injector.tool_failure import (
    ToolFailureConfig,
    ToolFailureInjector,
    ToolFailureType,
)


async def async_tool(x: int = 41) -> int:
    return x + 1


def sync_tool(x: int = 41) -> int:
    return x + 1


def _no_faults() -> ToolFailureInjector:
    return ToolFailureInjector(ToolFailureConfig(failure_type=ToolFailureType.NONE))


def _no_latency() -> LatencyFailureInjector:
    return LatencyFailureInjector(LatencyFailureConfig(failure_type=LatencyFailureType.NONE))


def test_call_async_returns_the_awaited_result() -> None:
    async def main() -> int:
        result = await _no_faults().call_async(async_tool)
        assert_type(result, int)
        return result + 1  # the reviewer's example: an int, not a coroutine

    assert asyncio.run(main()) == 43


def test_tool_injector_wrap_async_returns_a_coroutine_of_the_result() -> None:
    wrapped = _no_faults().wrap_async(async_tool)
    coroutine = wrapped()
    _ = assert_type(coroutine, Coroutine[Any, Any, int])
    coroutine.close()  # only checking its type here

    async def main() -> int:
        result = await wrapped(1)
        assert_type(result, int)
        return result

    assert asyncio.run(main()) == 2


def test_latency_injector_wrap_async_returns_a_coroutine_of_the_result() -> None:
    wrapped = _no_latency().wrap_async(async_tool)

    async def main() -> int:
        result = await wrapped()
        assert_type(result, int)
        return result

    assert asyncio.run(main()) == 42


def test_sync_wrappers_keep_the_return_type() -> None:
    tool_result = _no_faults().wrap(sync_tool)(1)
    assert_type(tool_result, int)
    assert tool_result == 2
    call_result = _no_faults().call(sync_tool)
    assert_type(call_result, int)
    assert call_result == 42
    latency_result = _no_latency().wrap(sync_tool)()
    assert_type(latency_result, int)
    assert latency_result == 42
