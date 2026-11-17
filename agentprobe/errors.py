"""Exceptions that describe the harness, not the agent under test.

Trials that end in one of these are recorded with ``status="harness_error"``
and excluded from every reliability metric, because they say nothing about
how the agent handles faults.
"""

from __future__ import annotations


class HarnessLimitation(Exception):
    """The harness cannot faithfully run this trial (an adapter limitation)."""


class ParallelToolCallsUnsupported(HarnessLimitation):
    """The model requested several tool calls in one turn, but the adapter can
    execute only one per turn (pyautogen 0.1.14's legacy ``function_call``).

    Running only the first call would silently change tool exposure and
    correctness, so the trial is rejected instead.
    """

    def __init__(self, n_calls: int, names: list[str]) -> None:
        self.n_calls = n_calls
        self.names = names
        super().__init__(
            f"Model requested {n_calls} parallel tool calls ({', '.join(names)}); "
            "the AutoGen adapter executes one tool call per turn"
        )


__all__ = ["HarnessLimitation", "ParallelToolCallsUnsupported"]
