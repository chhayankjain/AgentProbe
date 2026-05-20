"""Multi-agent benchmark task.

The MultiAgentTask presents a researcher + writer two-agent pipeline.
The researcher calls tools to gather information; the writer produces
a structured output. Used to benchmark:
  - Handoff failures between agents
  - State consistency across agent boundaries
  - Cascading failures when the researcher agent fails

The task inputs are static and self-contained — no external data required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MultiAgentQuery:
    query_id: str
    input: str
    researcher_tools: list[str]
    expected_output_keywords: list[str]


MULTI_AGENT_QUERIES: list[MultiAgentQuery] = [
    MultiAgentQuery(
        query_id="ma_001",
        input="Research the current state of LLM reliability in production and summarize findings.",
        researcher_tools=["web_search"],
        expected_output_keywords=["reliability", "production", "llm"],
    ),
    MultiAgentQuery(
        query_id="ma_002",
        input="Find the top 3 open-source agent frameworks and write a comparison.",
        researcher_tools=["web_search"],
        expected_output_keywords=["langgraph", "autogen", "comparison"],
    ),
    MultiAgentQuery(
        query_id="ma_003",
        input="Research observability tools for LLM systems and draft a feature comparison table.",
        researcher_tools=["web_search"],
        expected_output_keywords=["observability", "langsmith", "opentelemetry"],
    ),
    MultiAgentQuery(
        query_id="ma_004",
        input="Find recent benchmarks comparing GPT-4o and Claude Sonnet on coding tasks.",
        researcher_tools=["web_search"],
        expected_output_keywords=["gpt", "claude", "coding", "benchmark"],
    ),
    MultiAgentQuery(
        query_id="ma_005",
        input="Research MLSys 2025 workshop topics and draft a submission-ready abstract outline.",
        researcher_tools=["web_search"],
        expected_output_keywords=["mlsys", "workshop", "abstract"],
    ),
]


class MultiAgentTask:
    """Benchmark task: two-agent researcher + writer pipeline.

    The task tests handoff reliability and state propagation
    between agents in a sequential multi-agent workflow.
    """

    name = "multi_agent"
    description = "Two-agent researcher+writer pipeline benchmark"

    def __init__(self) -> None:
        self._queries = list(MULTI_AGENT_QUERIES)

    def get_inputs(self) -> list[dict[str, Any]]:
        return [
            {
                "input": q.input,
                "task": self.name,
                "query_id": q.query_id,
                "pipeline": "researcher->writer",
            }
            for q in self._queries
        ]

    def grade(self, query_id: str, output: str) -> bool:
        query = next((q for q in self._queries if q.query_id == query_id), None)
        if query is None:
            return False
        output_lower = output.lower()
        expected = query.expected_output_keywords
        return any(kw.lower() in output_lower for kw in expected)

    def __len__(self) -> int:
        return len(self._queries)
