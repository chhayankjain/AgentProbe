"""RAG (Retrieval-Augmented Generation) benchmark task.

The RAGTask presents the agent with questions that require retrieving
context from a document corpus before generating an answer. Used to
benchmark context-layer failures, including:
  - Context window overflow under long document injection
  - Lost state when retrieval history is dropped
  - Hallucinated history injection

The corpus is a static set of AgentProbe paper excerpts — fully
self-contained, no external API or database required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Static in-memory corpus (paper excerpts — reproducible, no external deps)
# ---------------------------------------------------------------------------

CORPUS: list[dict[str, str]] = [
    {
        "doc_id": "agentprobe_s1",
        "title": "Introduction",
        "content": (
            "Agentic LLM systems — systems where a language model iteratively "
            "calls tools, plans, and executes multi-step tasks — are rapidly "
            "moving from research demonstrations to production deployments. "
            "Unlike single-call LLM APIs, agentic systems introduce new failure "
            "modes at the orchestration, tool-use, and context layers that are "
            "poorly understood and under-instrumented."
        ),
    },
    {
        "doc_id": "agentprobe_s2",
        "title": "Failure Taxonomy",
        "content": (
            "We classify agentic failures into five categories: "
            "(1) Tool call failures — API timeouts, malformed outputs, rate limits; "
            "(2) Orchestration failures — infinite loops, wrong branching, state corruption; "
            "(3) Context failures — context window overflow, lost state, hallucinated history; "
            "(4) Latency failures — P99 spikes, cascading slowdowns, cold start delays; "
            "(5) Consistency failures — non-deterministic outputs, divergent multi-agent state."
        ),
    },
    {
        "doc_id": "agentprobe_s3",
        "title": "AgentProbe Design",
        "content": (
            "AgentProbe provides four core modules: a failure injector, an observability "
            "layer based on OpenTelemetry, a benchmark suite with standard tasks, and "
            "a metrics calculator. Reliability is measured using MTTR (mean time to "
            "recovery), failure rate, recovery rate, and P99 latency distributions."
        ),
    },
    {
        "doc_id": "agentprobe_s4",
        "title": "Experimental Setup",
        "content": (
            "Experiments compare LangGraph, LangChain, and AutoGen across three task types: "
            "tool-use agents, RAG agents, and multi-agent pipelines. LLMs used include "
            "Llama-3 via Ollama (local, zero API cost), GPT-4o, and Claude Sonnet. "
            "All experiments are reproducible from the public repository."
        ),
    },
    {
        "doc_id": "agentprobe_s5",
        "title": "Results",
        "content": (
            "LangGraph showed the lowest failure rate under tool timeout injection (12%) "
            "compared to LangChain (18%) and AutoGen (21%). Recovery rates were highest "
            "for LangGraph (87%) due to its explicit graph-based retry handling. "
            "P99 latency under cascading slowdown was 3.2x baseline for all frameworks."
        ),
    },
]


@dataclass
class RAGQuery:
    query_id: str
    input: str
    relevant_doc_ids: list[str]
    expected_keywords: list[str]


RAG_QUERIES: list[RAGQuery] = [
    RAGQuery(
        query_id="rag_001",
        input="What failure categories does AgentProbe define?",
        relevant_doc_ids=["agentprobe_s2"],
        expected_keywords=["tool", "orchestration", "context", "latency", "consistency"],
    ),
    RAGQuery(
        query_id="rag_002",
        input="How does AgentProbe measure agent reliability?",
        relevant_doc_ids=["agentprobe_s3"],
        expected_keywords=["mttr", "failure rate", "p99", "recovery"],
    ),
    RAGQuery(
        query_id="rag_003",
        input="Which agent framework had the lowest failure rate under timeout injection?",
        relevant_doc_ids=["agentprobe_s5"],
        expected_keywords=["langgraph", "12%"],
    ),
    RAGQuery(
        query_id="rag_004",
        input="What LLMs are used in the AgentProbe experiments?",
        relevant_doc_ids=["agentprobe_s4"],
        expected_keywords=["llama", "gpt", "claude", "ollama"],
    ),
    RAGQuery(
        query_id="rag_005",
        input="Why are agentic systems harder to observe than single-call LLM APIs?",
        relevant_doc_ids=["agentprobe_s1"],
        expected_keywords=["orchestration", "tool-use", "context", "failure"],
    ),
]


class RAGTask:
    """Benchmark task: RAG agent with static in-memory corpus.

    No external vector database required — retrieval is keyword-based over
    the static CORPUS for fully reproducible benchmarking.
    """

    name = "rag"
    description = "RAG agent benchmark with static document corpus"

    def __init__(self) -> None:
        self._queries = list(RAG_QUERIES)
        self._corpus = {doc["doc_id"]: doc for doc in CORPUS}

    def retrieve(self, query: str, top_k: int = 2) -> list[dict[str, str]]:
        """Simple keyword-overlap retrieval (no embeddings required)."""
        query_tokens = set(query.lower().split())
        scored: list[tuple[float, dict[str, str]]] = []
        for doc in self._corpus.values():
            doc_tokens = set(doc["content"].lower().split())
            overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
            scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def get_inputs(self) -> list[dict[str, Any]]:
        """Return task inputs with pre-retrieved context."""
        inputs = []
        for q in self._queries:
            context_docs = self.retrieve(q.input)
            context = "\n\n".join(
                f"[{d['title']}]: {d['content']}" for d in context_docs
            )
            inputs.append({
                "input": q.input,
                "context": context,
                "task": self.name,
                "query_id": q.query_id,
            })
        return inputs

    def grade(self, query_id: str, output: str) -> bool:
        query = next((q for q in self._queries if q.query_id == query_id), None)
        if query is None:
            return False
        output_lower = output.lower()
        return any(kw.lower() in output_lower for kw in query.expected_keywords)

    def __len__(self) -> int:
        return len(self._queries)
