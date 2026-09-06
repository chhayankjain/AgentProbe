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
    # --- Expanded queries (ma_006 – ma_015) ---
    MultiAgentQuery(
        query_id="ma_006",
        input=(
            "Research the latest advances in protein structure prediction "
            "and write a technical blog post suitable for a machine learning audience."
        ),
        researcher_tools=["web_search"],
        expected_output_keywords=["protein", "alphafold", "structure", "prediction"],
    ),
    MultiAgentQuery(
        query_id="ma_007",
        input=(
            "Analyze the S&P 500 performance over the last quarter "
            "and produce a risk assessment report with key takeaways."
        ),
        researcher_tools=["web_search", "calculator"],
        expected_output_keywords=["s&p", "return", "risk", "quarter"],
    ),
    MultiAgentQuery(
        query_id="ma_008",
        input=(
            "Research climate change mitigation strategies in urban areas "
            "and draft a policy brief with three actionable recommendations."
        ),
        researcher_tools=["web_search"],
        expected_output_keywords=["climate", "urban", "mitigation", "recommendation"],
    ),
    MultiAgentQuery(
        query_id="ma_009",
        input=(
            "Find the top 5 most cited papers on prompt engineering in 2024 "
            "and write a literature review section summarizing their contributions."
        ),
        researcher_tools=["web_search"],
        expected_output_keywords=["prompt", "engineering", "cited", "review"],
    ),
    MultiAgentQuery(
        query_id="ma_010",
        input=(
            "Research the trade-offs between fine-tuning and retrieval-augmented "
            "generation for domain-specific tasks, then produce a decision matrix "
            "that a practitioner could use to choose between them."
        ),
        researcher_tools=["web_search"],
        expected_output_keywords=["fine-tuning", "rag", "trade-off", "decision"],
    ),
    MultiAgentQuery(
        query_id="ma_011",
        input=(
            "One agent should propose a software architecture for a real-time "
            "fraud detection system, and another agent should critique the design "
            "and suggest improvements."
        ),
        researcher_tools=["web_search"],
        expected_output_keywords=["fraud", "architecture", "critique", "improvement"],
    ),
    MultiAgentQuery(
        query_id="ma_012",
        input=(
            "Plan a three-phase rollout strategy for deploying an LLM-powered "
            "customer support chatbot, then write a risk register for each phase."
        ),
        researcher_tools=["web_search"],
        expected_output_keywords=["rollout", "phase", "risk", "chatbot"],
    ),
    MultiAgentQuery(
        query_id="ma_013",
        input=(
            "Research the energy consumption of training large language models "
            "and draft an executive summary with both environmental costs "
            "and proposed efficiency improvements."
        ),
        researcher_tools=["web_search", "calculator"],
        expected_output_keywords=["energy", "training", "efficiency", "carbon"],
    ),
    MultiAgentQuery(
        query_id="ma_014",
        input=(
            "A researcher agent must gather data on global semiconductor supply "
            "chain disruptions in 2024, while a writer agent must produce a "
            "concise briefing document with sourced claims. The writer should "
            "flag any claims that lack sufficient evidence."
        ),
        researcher_tools=["web_search"],
        expected_output_keywords=["semiconductor", "supply chain", "disruption", "source"],
    ),
    MultiAgentQuery(
        query_id="ma_015",
        input=(
            "Research both the benefits and risks of autonomous AI agents "
            "in healthcare diagnostics, then produce a balanced position paper "
            "with arguments for and against deployment."
        ),
        researcher_tools=["web_search"],
        expected_output_keywords=["healthcare", "autonomous", "benefit", "risk"],
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
