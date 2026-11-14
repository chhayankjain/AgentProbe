"""Standard benchmark task corpora for AgentProbe.

Tasks are designed to be:
  1. Reproducible — deterministic inputs, public data sources
  2. Comparable — same inputs applied across all frameworks
  3. Realistic — representative of production agentic use cases

Task categories:
  - tool_use     : Single-agent, multi-tool calling (25 queries)
  - rag          : Retrieval-augmented generation over a static corpus (15 queries)
  - multi_agent  : Two-agent researcher + writer pipeline (15 queries)

**Relationship to the paper.** The experiments reported in the paper did *not*
use these corpora. They used the five-prompt-per-task set in
:data:`agentprobe.benchmark.runner.PAPER_PROMPTS`, which is also the
:class:`~agentprobe.benchmark.runner.BenchmarkRunner` default so that the
published batches reproduce exactly. To benchmark against the larger corpora
instead, pass :func:`corpus_task_fn` as the runner's ``task_fn``::

    from agentprobe.benchmark import BenchmarkRunner, BenchmarkConfig, corpus_task_fn

    runner = BenchmarkRunner(
        agent=agent,
        config=BenchmarkConfig(task="tool_use", n_runs=50),
        task_fn=corpus_task_fn("tool_use"),
    )
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agentprobe.benchmark.tasks.multi_agent import MultiAgentTask
from agentprobe.benchmark.tasks.rag import RAGTask
from agentprobe.benchmark.tasks.tool_use import ToolUseTask

__all__ = ["ToolUseTask", "RAGTask", "MultiAgentTask", "get_task", "corpus_task_fn"]

_TASK_CLASSES: dict[str, type] = {
    ToolUseTask.name: ToolUseTask,
    RAGTask.name: RAGTask,
    MultiAgentTask.name: MultiAgentTask,
}


def get_task(name: str) -> ToolUseTask | RAGTask | MultiAgentTask:
    """Return a task instance by name (``"tool_use"``, ``"rag"``, ``"multi_agent"``)."""
    try:
        cls = _TASK_CLASSES[name]
    except KeyError:
        raise ValueError(
            f"Unknown task {name!r}; expected one of {sorted(_TASK_CLASSES)}"
        ) from None
    return cls()  # type: ignore[no-any-return]


def corpus_task_fn(name: str) -> Callable[[int, str], dict[str, Any]]:
    """Build a ``task_fn`` for :class:`~agentprobe.benchmark.runner.BenchmarkRunner`
    that cycles through the full query corpus for *name*.

    Run ``i`` receives query ``i % len(corpus)``. The returned dict carries the
    ``query_id`` so outputs can be graded with ``get_task(name).grade(...)``.
    """
    task = get_task(name)
    inputs = task.get_inputs()
    if not inputs:
        raise ValueError(f"Task {name!r} has no queries")

    def _task_fn(run_id: int, _task_name: str) -> dict[str, Any]:
        item = dict(inputs[run_id % len(inputs)])
        item["run_id"] = run_id
        return item

    return _task_fn
