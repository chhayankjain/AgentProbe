"""Run blocking work in a thread that never keeps the process alive."""

from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import Callable
from typing import Any


def start_daemon(
    fn: Callable[[], Any], name: str = "agentprobe-agent"
) -> concurrent.futures.Future[Any]:
    """Run *fn* in a daemon thread and return a Future for its outcome.

    ``ThreadPoolExecutor`` and ``asyncio.to_thread`` workers are joined when the
    interpreter exits, so an agent that hangs past its deadline would keep a test
    run alive until it finished. Python cannot stop a thread; a daemon thread at
    least lets the process exit.
    """
    future: concurrent.futures.Future[Any] = concurrent.futures.Future()

    def runner() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = fn()
        except BaseException as exc:  # noqa: BLE001 - delivered to the waiter
            future.set_exception(exc)
        else:
            future.set_result(result)

    threading.Thread(target=runner, name=name, daemon=True).start()
    return future
