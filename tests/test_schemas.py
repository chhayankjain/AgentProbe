"""Tests for Chunk 1 — Config, injector types, benchmark types, and agent base."""

from __future__ import annotations

import os
from datetime import datetime

import pytest
from pydantic import ValidationError

from agentprobe.agents.base import AgentResult, BaseAgent
from agentprobe.benchmark.types import (
    AgentFramework,
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkResults,
    RunResult,
    TaskType,
)
from agentprobe.config import Settings
from agentprobe.injector.types import ToolFailureConfig, ToolFailureType


class TestToolFailureConfig:
    def test_default_is_no_op(self):
        cfg = ToolFailureConfig()
        assert cfg.failure_type == ToolFailureType.NONE
        assert cfg.failure_probability == 0.0
        assert cfg.max_retries == 3
        assert cfg.seed is None

    def test_full_construction(self):
        cfg = ToolFailureConfig(
            failure_type=ToolFailureType.TIMEOUT,
            failure_probability=0.5,
            timeout_seconds=10.0,
            seed=42,
            max_retries=1,
        )
        assert cfg.failure_type == ToolFailureType.TIMEOUT
        assert cfg.seed == 42

    def test_probability_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            ToolFailureConfig(failure_probability=-0.1)

    def test_probability_above_one_rejected(self):
        with pytest.raises(ValidationError):
            ToolFailureConfig(failure_probability=1.1)

    def test_probability_boundary_values_accepted(self):
        ToolFailureConfig(failure_probability=0.0)
        ToolFailureConfig(failure_probability=1.0)

    def test_timeout_seconds_must_be_positive(self):
        with pytest.raises(ValidationError):
            ToolFailureConfig(timeout_seconds=0.0)

    def test_all_failure_types_are_valid(self):
        for ft in ToolFailureType:
            cfg = ToolFailureConfig(failure_type=ft)
            assert cfg.failure_type == ft

    def test_round_trip_serialization(self):
        cfg = ToolFailureConfig(
            failure_type=ToolFailureType.RATE_LIMIT,
            failure_probability=0.3,
            seed=7,
        )
        reloaded = ToolFailureConfig.model_validate(cfg.model_dump())
        assert reloaded == cfg


class TestBenchmarkConfig:
    def test_minimal_construction(self):
        cfg = BenchmarkConfig(
            framework=AgentFramework.LANGGRAPH,
            task=TaskType.TOOL_USE,
        )
        assert cfg.n_runs == 10
        assert cfg.recovery_attempts == 3
        assert cfg.save_results_path is None

    def test_full_construction(self):
        failure_cfg = ToolFailureConfig(
            failure_type=ToolFailureType.TIMEOUT,
            failure_probability=0.2,
            seed=42,
        )
        cfg = BenchmarkConfig(
            framework=AgentFramework.LANGCHAIN,
            task=TaskType.RAG,
            n_runs=50,
            failure_config=failure_cfg,
            recovery_attempts=1,
            save_results_path="experiments/results/run_01",
        )
        assert cfg.n_runs == 50
        assert cfg.failure_config.seed == 42

    def test_n_runs_must_be_at_least_one(self):
        with pytest.raises(ValidationError):
            BenchmarkConfig(
                framework=AgentFramework.LANGGRAPH,
                task=TaskType.TOOL_USE,
                n_runs=0,
            )

    def test_round_trip_serialization(self):
        cfg = BenchmarkConfig(
            framework=AgentFramework.AUTOGEN,
            task=TaskType.MULTI_AGENT,
            n_runs=5,
        )
        reloaded = BenchmarkConfig.model_validate(cfg.model_dump())
        assert reloaded == cfg


class TestRunResult:
    def _make(self, **kwargs: object) -> RunResult:
        defaults: dict[str, object] = dict(
            run_id="run-001",
            framework=AgentFramework.LANGGRAPH,
            task=TaskType.TOOL_USE,
            success=True,
            failure_type=ToolFailureType.NONE,
            duration_ms=250.0,
        )
        defaults.update(kwargs)
        return RunResult(**defaults)  # type: ignore[arg-type]

    def test_minimal_success_run(self):
        r = self._make()
        assert r.recovery_time_ms is None
        assert r.retries == 0
        assert r.error_message is None
        assert isinstance(r.timestamp, datetime)

    def test_failed_run_with_recovery(self):
        r = self._make(
            success=False,
            failure_type=ToolFailureType.TIMEOUT,
            recovery_time_ms=120.0,
            retries=2,
            error_message="Tool call timed out",
        )
        assert not r.success
        assert r.recovery_time_ms == 120.0

    def test_round_trip_serialization(self):
        r = self._make(failure_type=ToolFailureType.API_ERROR, success=False)
        reloaded = RunResult.model_validate(r.model_dump())
        assert reloaded.run_id == r.run_id
        assert reloaded.failure_type == ToolFailureType.API_ERROR


class TestBenchmarkResults:
    def test_empty_runs(self):
        cfg = BenchmarkConfig(
            framework=AgentFramework.LANGGRAPH,
            task=TaskType.TOOL_USE,
        )
        results = BenchmarkResults(config=cfg, runs=[], total_duration_ms=0.0)
        assert results.runs == []


class TestBenchmarkMetrics:
    def test_construction(self):
        m = BenchmarkMetrics(
            failure_rate=0.2,
            mttr_ms=500.0,
            p99_latency_ms=1200.0,
            p50_latency_ms=300.0,
            total_runs=50,
            successful_runs=40,
        )
        assert m.failure_rate == 0.2
        assert m.successful_runs == 40


class TestAgentResult:
    def test_successful_result(self):
        r = AgentResult(success=True, output="42", duration_ms=100.0)
        assert r.retries == 0
        assert r.error_message is None

    def test_failed_result(self):
        r = AgentResult(
            success=False,
            duration_ms=50.0,
            retries=3,
            error_message="LLM returned empty output",
        )
        assert r.output is None

    def test_duration_cannot_be_negative(self):
        with pytest.raises(ValidationError):
            AgentResult(success=True, duration_ms=-1.0)


class TestBaseAgentProtocol:
    def test_protocol_is_runtime_checkable(self):
        class StubAgent:
            def run(self, task_input: str) -> AgentResult:
                return AgentResult(success=True, output="ok", duration_ms=10.0)

            def reset(self) -> None:
                pass

        agent = StubAgent()
        assert isinstance(agent, BaseAgent)

    def test_incomplete_class_fails_isinstance(self):
        class IncompleteAgent:
            def run(self, task_input: str) -> AgentResult:
                return AgentResult(success=True, duration_ms=0.0)

            # missing reset()

        assert not isinstance(IncompleteAgent(), BaseAgent)


class TestSettings:
    def test_defaults_when_env_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("AGENTPROBE_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("AGENTPROBE_NO_CONSOLE_TRACE", raising=False)
        monkeypatch.delenv("AGENTPROBE_PROMETHEUS_PORT", raising=False)

        s = Settings.from_env()
        assert s.openai_api_key is None
        assert s.anthropic_api_key is None
        assert s.ollama_base_url == "http://localhost:11434"
        assert s.otlp_endpoint is None
        assert s.no_console_trace is False
        assert s.prometheus_port == 8000

    def test_reads_openai_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        s = Settings.from_env()
        assert s.openai_api_key == "sk-test-123"

    def test_require_openai_raises_when_absent(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        s = Settings.from_env()
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            s.require_openai()

    def test_require_openai_returns_key_when_present(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-abc")
        s = Settings.from_env()
        assert s.require_openai() == "sk-abc"

    def test_require_anthropic_raises_when_absent(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        s = Settings.from_env()
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            s.require_anthropic()

    def test_no_console_trace_flag(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AGENTPROBE_NO_CONSOLE_TRACE", "1")
        s = Settings.from_env()
        assert s.no_console_trace is True

    def test_prometheus_port_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AGENTPROBE_PROMETHEUS_PORT", "9090")
        s = Settings.from_env()
        assert s.prometheus_port == 9090
