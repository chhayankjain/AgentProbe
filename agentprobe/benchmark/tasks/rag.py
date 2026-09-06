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
    # --- Expanded corpus documents (s6 – s10) ---
    {
        "doc_id": "agentprobe_s6",
        "title": "Fault Injection Methodology",
        "content": (
            "Fault injection in AgentProbe operates at the tool-call boundary. "
            "Injectors wrap tool functions with configurable failure modes: timeout "
            "delays, malformed JSON responses, HTTP 429 rate-limit errors, and "
            "generic API errors. Each injector is stateless and composable, allowing "
            "multiple failure modes to be layered in a single experiment run. "
            "Injection probability is configurable from 0% to 100%."
        ),
    },
    {
        "doc_id": "agentprobe_s7",
        "title": "Observability Patterns",
        "content": (
            "The observability layer exports OpenTelemetry spans for every agent step, "
            "tool invocation, and LLM call. Spans carry structured attributes including "
            "failure type, retry count, token usage, and latency. A Prometheus exporter "
            "aggregates step-level metrics into counters, histograms, and gauges. "
            "Grafana dashboards visualize failure rates, MTTR trends, and P99 latency "
            "distributions in real time."
        ),
    },
    {
        "doc_id": "agentprobe_s8",
        "title": "LLM Agent Architectures",
        "content": (
            "Three agent architectures are evaluated: LangGraph uses a finite state "
            "machine with explicit edges and conditional branching. LangChain uses a "
            "ReAct-style agent loop with an AgentExecutor that iterates until a final "
            "answer is produced. AutoGen uses a conversational multi-agent protocol "
            "where agents exchange messages in a group chat. Each architecture handles "
            "tool failures differently: LangGraph retries via graph edges, LangChain "
            "relies on the LLM to self-correct, and AutoGen delegates to peer agents."
        ),
    },
    {
        "doc_id": "agentprobe_s9",
        "title": "Chaos Engineering for AI",
        "content": (
            "Chaos engineering principles — originally developed for distributed systems "
            "by Netflix — are adapted for agentic AI. The steady-state hypothesis is "
            "defined as the agent completing its task within a latency budget and "
            "producing a correct answer. Perturbations include tool failures, context "
            "window exhaustion, and inter-agent message loss. Blast radius is limited "
            "by running experiments in isolated sandbox environments with mock tools."
        ),
    },
    {
        "doc_id": "agentprobe_s10",
        "title": "Benchmark Design Principles",
        "content": (
            "Benchmark tasks are designed for reproducibility, coverage, and minimal "
            "external dependencies. Each task provides static inputs, deterministic "
            "grading via keyword matching, and pre-defined tool schemas. The three "
            "task categories — tool-use, RAG, and multi-agent — cover single-agent "
            "tool invocation, retrieval-augmented generation with a static corpus, "
            "and multi-agent coordination pipelines respectively. Task difficulty is "
            "controlled by the number of required tool calls and reasoning steps."
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
    # --- Expanded queries (rag_006 – rag_015) ---
    RAGQuery(
        query_id="rag_006",
        input="How does AgentProbe inject faults at the tool-call boundary?",
        relevant_doc_ids=["agentprobe_s6"],
        expected_keywords=["timeout", "malformed", "rate-limit", "injection"],
    ),
    RAGQuery(
        query_id="rag_007",
        input="What observability signals does AgentProbe export via OpenTelemetry?",
        relevant_doc_ids=["agentprobe_s7"],
        expected_keywords=["spans", "opentelemetry", "prometheus", "latency"],
    ),
    RAGQuery(
        query_id="rag_008",
        input="Compare how LangGraph, LangChain, and AutoGen handle tool failures.",
        relevant_doc_ids=["agentprobe_s8", "agentprobe_s5"],
        expected_keywords=["retry", "langgraph", "langchain", "autogen"],
    ),
    RAGQuery(
        query_id="rag_009",
        input="How are chaos engineering principles adapted for agentic AI systems?",
        relevant_doc_ids=["agentprobe_s9"],
        expected_keywords=["chaos", "steady-state", "perturbation", "blast radius"],
    ),
    RAGQuery(
        query_id="rag_010",
        input="What design principles guide the AgentProbe benchmark tasks?",
        relevant_doc_ids=["agentprobe_s10"],
        expected_keywords=["reproducibility", "coverage", "keyword", "grading"],
    ),
    RAGQuery(
        query_id="rag_011",
        input=(
            "Synthesize the fault injection methodology and observability patterns: "
            "how do injected faults become visible in the monitoring stack?"
        ),
        relevant_doc_ids=["agentprobe_s6", "agentprobe_s7"],
        expected_keywords=["injection", "spans", "failure type", "prometheus"],
    ),
    RAGQuery(
        query_id="rag_012",
        input="What is the injection probability range supported by AgentProbe?",
        relevant_doc_ids=["agentprobe_s6"],
        expected_keywords=["0%", "100%", "configurable"],
    ),
    RAGQuery(
        query_id="rag_013",
        input=(
            "Which agent architecture uses a finite state machine, and what advantage "
            "does this provide for retry handling?"
        ),
        relevant_doc_ids=["agentprobe_s8"],
        expected_keywords=["langgraph", "state machine", "edges", "retry"],
    ),
    RAGQuery(
        query_id="rag_014",
        input=(
            "How does AgentProbe limit the blast radius of chaos experiments, "
            "and why is this important?"
        ),
        relevant_doc_ids=["agentprobe_s9"],
        expected_keywords=["sandbox", "mock", "isolated", "blast radius"],
    ),
    RAGQuery(
        query_id="rag_015",
        input=(
            "Across all sections, what metrics does AgentProbe use to quantify "
            "agent reliability and performance?"
        ),
        relevant_doc_ids=["agentprobe_s3", "agentprobe_s7"],
        expected_keywords=["mttr", "failure rate", "p99", "histogram"],
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
