"""Tool-use benchmark task definition.

The ToolUseTask presents the agent with queries that require invoking
one or more tools to produce a correct answer. Used to benchmark:
  - Tool call failure rates under injection
  - Recovery behavior after tool timeouts/errors
  - Latency distribution for multi-tool invocations

All queries are factual and verifiable — no subjective evaluation needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolUseQuery:
    query_id: str
    input: str
    required_tools: list[str]
    expected_answer_keywords: list[str]  # simple keyword-based grading


# Reproducible query set — static, no external data dependency
TOOL_USE_QUERIES: list[ToolUseQuery] = [
    ToolUseQuery(
        query_id="tu_001",
        input="What is the current population of Tokyo?",
        required_tools=["web_search"],
        expected_answer_keywords=["million", "tokyo", "population"],
    ),
    ToolUseQuery(
        query_id="tu_002",
        input="Calculate the compound interest on $10,000 at 5% annually for 3 years.",
        required_tools=["calculator"],
        expected_answer_keywords=["11576", "interest", "compound"],
    ),
    ToolUseQuery(
        query_id="tu_003",
        input="What are the top 3 LLM papers published in 2024 on arXiv?",
        required_tools=["web_search"],
        expected_answer_keywords=["arxiv", "2024", "llm"],
    ),
    ToolUseQuery(
        query_id="tu_004",
        input="Convert 100 USD to EUR using the current exchange rate.",
        required_tools=["web_search"],
        expected_answer_keywords=["eur", "exchange", "rate"],
    ),
    ToolUseQuery(
        query_id="tu_005",
        input="What is the capital of the country with the highest GDP in Southeast Asia?",
        required_tools=["web_search"],
        expected_answer_keywords=["jakarta", "singapore", "bangkok", "capital"],
    ),
    ToolUseQuery(
        query_id="tu_006",
        input="Summarize the abstract of the LangGraph paper.",
        required_tools=["web_search"],
        expected_answer_keywords=["langgraph", "graph", "state"],
    ),
    ToolUseQuery(
        query_id="tu_007",
        input="What Python version introduced structural pattern matching?",
        required_tools=["web_search"],
        expected_answer_keywords=["3.10", "python", "match"],
    ),
    ToolUseQuery(
        query_id="tu_008",
        input="Compute 2 to the power of 32.",
        required_tools=["calculator"],
        expected_answer_keywords=["4294967296"],
    ),
    ToolUseQuery(
        query_id="tu_009",
        input="Who won the Turing Award in 2023?",
        required_tools=["web_search"],
        expected_answer_keywords=["turing", "award", "2023"],
    ),
    ToolUseQuery(
        query_id="tu_010",
        input="List the top 5 open-source LLMs by HuggingFace downloads in 2024.",
        required_tools=["web_search"],
        expected_answer_keywords=["llama", "mistral", "huggingface"],
    ),
    # --- Expanded queries (tu_011 – tu_025) ---
    ToolUseQuery(
        query_id="tu_011",
        input="What is the molecular weight of caffeine in g/mol?",
        required_tools=["web_search"],
        expected_answer_keywords=["194", "caffeine", "molecular"],
    ),
    ToolUseQuery(
        query_id="tu_012",
        input="Calculate the factorial of 12.",
        required_tools=["calculator"],
        expected_answer_keywords=["479001600"],
    ),
    ToolUseQuery(
        query_id="tu_013",
        input="What is the distance in kilometers between the Earth and Mars at closest approach?",
        required_tools=["web_search"],
        expected_answer_keywords=["million", "km", "mars"],
    ),
    ToolUseQuery(
        query_id="tu_014",
        input="Calculate the monthly payment on a $300,000 mortgage at 6.5% APR over 30 years.",
        required_tools=["calculator"],
        expected_answer_keywords=["1896", "monthly", "payment"],
    ),
    ToolUseQuery(
        query_id="tu_015",
        input="What is the GDP per capita of Switzerland in 2024 USD?",
        required_tools=["web_search"],
        expected_answer_keywords=["switzerland", "gdp", "per capita"],
    ),
    ToolUseQuery(
        query_id="tu_016",
        input="Search for the boiling point of ethanol and then convert it from Celsius to Fahrenheit.",
        required_tools=["web_search", "calculator"],
        expected_answer_keywords=["173", "ethanol", "fahrenheit"],
    ),
    ToolUseQuery(
        query_id="tu_017",
        input="What programming language was used to write the first version of Git?",
        required_tools=["web_search"],
        expected_answer_keywords=["c", "git", "torvalds"],
    ),
    ToolUseQuery(
        query_id="tu_018",
        input="Calculate the area of a circle with radius 7.5 meters.",
        required_tools=["calculator"],
        expected_answer_keywords=["176.7", "area", "circle"],
    ),
    ToolUseQuery(
        query_id="tu_019",
        input="What year did the Byzantine Empire fall, and what was the conquering force?",
        required_tools=["web_search"],
        expected_answer_keywords=["1453", "ottoman", "constantinople"],
    ),
    ToolUseQuery(
        query_id="tu_020",
        input="Find the current price of gold per troy ounce in USD and calculate the value of 3.5 ounces.",
        required_tools=["web_search", "calculator"],
        expected_answer_keywords=["gold", "ounce", "usd"],
    ),
    ToolUseQuery(
        query_id="tu_021",
        input="What is the deepest point in the ocean and how deep is it in meters?",
        required_tools=["web_search"],
        expected_answer_keywords=["mariana", "challenger", "10"],
    ),
    ToolUseQuery(
        query_id="tu_022",
        input="Calculate the sum of all prime numbers less than 50.",
        required_tools=["calculator"],
        expected_answer_keywords=["328"],
    ),
    ToolUseQuery(
        query_id="tu_023",
        input="Which country has the most UNESCO World Heritage Sites as of 2024?",
        required_tools=["web_search"],
        expected_answer_keywords=["italy", "china", "heritage"],
    ),
    ToolUseQuery(
        query_id="tu_024",
        input=(
            "Look up the population of both India and China, "
            "then calculate which country has a higher population density "
            "given India is 3.287 million km2 and China is 9.597 million km2."
        ),
        required_tools=["web_search", "calculator"],
        expected_answer_keywords=["india", "density", "higher"],
    ),
    ToolUseQuery(
        query_id="tu_025",
        input="What is the time complexity of merge sort and why is it considered efficient?",
        required_tools=["web_search"],
        expected_answer_keywords=["n log n", "merge", "sort", "divide"],
    ),
]


class ToolUseTask:
    """Benchmark task: tool-use agent.

    Provides a fixed set of queries that require tool invocations.
    Includes a simple keyword-based grader for automated evaluation.
    """

    name = "tool_use"
    description = "Single-agent, multi-tool benchmark requiring web search and calculation"

    def __init__(self, query_ids: list[str] | None = None) -> None:
        if query_ids:
            self._queries = [q for q in TOOL_USE_QUERIES if q.query_id in query_ids]
        else:
            self._queries = list(TOOL_USE_QUERIES)

    def get_inputs(self) -> list[dict[str, Any]]:
        """Return task inputs suitable for agent.invoke()."""
        return [
            {
                "input": q.input,
                "task": self.name,
                "query_id": q.query_id,
            }
            for q in self._queries
        ]

    def grade(self, query_id: str, output: str) -> bool:
        """Return True if the output contains expected keywords (case-insensitive)."""
        query = next((q for q in self._queries if q.query_id == query_id), None)
        if query is None:
            return False
        output_lower = output.lower()
        return any(kw.lower() in output_lower for kw in query.expected_answer_keywords)

    def __len__(self) -> int:
        return len(self._queries)
