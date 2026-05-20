"""Tests for benchmark task definitions — tool_use, rag, multi_agent."""

from __future__ import annotations

import pytest

from agentprobe.benchmark.tasks.tool_use import ToolUseTask, TOOL_USE_QUERIES
from agentprobe.benchmark.tasks.rag import RAGTask, RAG_QUERIES, CORPUS
from agentprobe.benchmark.tasks.multi_agent import MultiAgentTask, MULTI_AGENT_QUERIES


class TestToolUseTask:
    def test_default_has_all_queries(self):
        task = ToolUseTask()
        assert len(task) == len(TOOL_USE_QUERIES)

    def test_subset_by_query_ids(self):
        task = ToolUseTask(query_ids=["tu_001", "tu_002"])
        assert len(task) == 2

    def test_get_inputs_returns_list_of_dicts(self):
        task = ToolUseTask()
        inputs = task.get_inputs()
        assert len(inputs) == len(task)
        for inp in inputs:
            assert "input" in inp
            assert "query_id" in inp
            assert "task" in inp
            assert inp["task"] == "tool_use"

    def test_grade_keyword_match(self):
        task = ToolUseTask()
        assert task.grade("tu_001", "Tokyo population is 14 million people.")

    def test_grade_case_insensitive(self):
        task = ToolUseTask()
        assert task.grade("tu_001", "TOKYO HAS MILLIONS OF RESIDENTS")

    def test_grade_no_match_returns_false(self):
        task = ToolUseTask()
        assert not task.grade("tu_001", "I have no idea what you are asking.")

    def test_grade_unknown_query_id_returns_false(self):
        task = ToolUseTask()
        assert not task.grade("nonexistent_id", "Tokyo population")

    def test_all_queries_have_required_tools(self):
        for q in TOOL_USE_QUERIES:
            assert len(q.required_tools) >= 1

    def test_calculator_query_grading(self):
        task = ToolUseTask()
        assert task.grade("tu_008", "2^32 = 4294967296")


class TestRAGTask:
    def test_has_correct_query_count(self):
        task = RAGTask()
        assert len(task) == len(RAG_QUERIES)

    def test_corpus_has_documents(self):
        assert len(CORPUS) >= 3

    def test_retrieve_returns_top_k_docs(self):
        task = RAGTask()
        docs = task.retrieve("failure categories AgentProbe", top_k=2)
        assert len(docs) == 2
        for doc in docs:
            assert "doc_id" in doc
            assert "content" in doc
            assert "title" in doc

    def test_retrieve_default_top_k_is_2(self):
        task = RAGTask()
        docs = task.retrieve("tool call failures")
        assert len(docs) == 2

    def test_get_inputs_includes_context(self):
        task = RAGTask()
        inputs = task.get_inputs()
        for inp in inputs:
            assert "context" in inp
            assert len(inp["context"]) > 0
            assert "task" in inp
            assert inp["task"] == "rag"

    def test_grade_relevant_answer(self):
        task = RAGTask()
        # rag_001: failure categories
        assert task.grade(
            "rag_001",
            "AgentProbe defines tool call, orchestration, context, latency, and consistency failures."
        )

    def test_grade_wrong_answer(self):
        task = RAGTask()
        assert not task.grade("rag_001", "I cannot answer that question.")

    def test_retrieval_relevance_for_taxonomy_query(self):
        task = RAGTask()
        docs = task.retrieve("failure taxonomy orchestration tool context")
        # The taxonomy document should be in top results
        doc_ids = [d["doc_id"] for d in docs]
        assert "agentprobe_s2" in doc_ids


class TestMultiAgentTask:
    def test_has_correct_query_count(self):
        task = MultiAgentTask()
        assert len(task) == len(MULTI_AGENT_QUERIES)

    def test_get_inputs_returns_dicts(self):
        task = MultiAgentTask()
        inputs = task.get_inputs()
        assert len(inputs) == len(task)
        for inp in inputs:
            assert "input" in inp
            assert "query_id" in inp
            assert "pipeline" in inp

    def test_pipeline_field_is_correct(self):
        task = MultiAgentTask()
        for inp in task.get_inputs():
            assert inp["pipeline"] == "researcher->writer"

    def test_grade_keyword_match(self):
        task = MultiAgentTask()
        assert task.grade("ma_001", "LLM reliability in production systems")

    def test_grade_no_match(self):
        task = MultiAgentTask()
        assert not task.grade("ma_001", "Completely unrelated response about nothing.")

    def test_all_queries_have_researcher_tools(self):
        for q in MULTI_AGENT_QUERIES:
            assert len(q.researcher_tools) >= 1

    def test_all_queries_have_expected_keywords(self):
        for q in MULTI_AGENT_QUERIES:
            assert len(q.expected_output_keywords) >= 1
