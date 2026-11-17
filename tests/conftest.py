"""Shared fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_autogen_shim_state():
    """The AutoGen shim keeps per-thread throttle/retry/cache state. Tests run in
    one thread, so reset it around every test to keep tests independent."""
    try:
        from agentprobe.agents import autogen_agent as ag
    except ImportError:  # pragma: no cover
        yield
        return
    ag._activate(None, None)
    yield
    ag._activate(None, None)
