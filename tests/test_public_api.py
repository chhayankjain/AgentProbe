"""Public API contract tests.

Guards the things a first-time user touches: top-level exports, the
``ToolFailureConfig`` that the injector actually accepts, the ``BaseAgent``
protocol that the adapters actually satisfy, task-corpus helpers, and
``Settings``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

import agentprobe
from agentprobe.agents.base import AgentResult, BaseAgent
from agentprobe.benchmark import PAPER_PROMPTS, corpus_task_fn, get_task, paper_task_fn
from agentprobe.config import Settings
from agentprobe.injector import ToolFailureConfig, ToolFailureInjector, ToolFailureType
from agentprobe.injector import tool_failure as _tool_failure_module


class TestTopLevelExports:
    """The README's import lines must work."""

    def test_readme_imports(self):
        from agentprobe import (  # noqa: F401
            AgentProbeTracer,
            BenchmarkConfig,
            BenchmarkRunner,
            MetricsCalculator,
            ToolFailureConfig,
            ToolFailureInjector,
            ToolFailureType,
        )

    def test_all_names_resolve(self):
        for name in agentprobe.__all__:
            assert hasattr(agentprobe, name), name

    def test_version_consistent_across_release_files(self):
        """RELEASE.md says the version lives in three places; keep them in sync."""
        import re
        import tomllib
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
        assert project["name"] == "agentprobe-toolkit"
        assert agentprobe.__version__ == project["version"]
        cff = (root / "CITATION.cff").read_text()
        m = re.search(r'^version:\s*"([^"]+)"', cff, re.MULTILINE)
        assert m and m.group(1) == project["version"], "CITATION.cff version out of sync"


class TestSingleToolFailureConfig:
    """There is exactly one ToolFailureConfig / ToolFailureType, and the
    injector accepts the one the package exports."""

    def test_package_export_is_the_injector_config(self):
        assert ToolFailureConfig is _tool_failure_module.ToolFailureConfig
        assert ToolFailureType is _tool_failure_module.ToolFailureType
        assert agentprobe.ToolFailureConfig is ToolFailureConfig

    def test_legacy_types_module_is_gone(self):
        with pytest.raises(ImportError):
            import agentprobe.injector.types  # noqa: F401
        with pytest.raises(ImportError):
            import agentprobe.benchmark.types  # noqa: F401

    def test_default_is_no_op(self):
        cfg = ToolFailureConfig()
        assert cfg.failure_type == ToolFailureType.NONE
        assert cfg.seed is None
        assert ToolFailureInjector(cfg).call(lambda: "ok") == "ok"

    def test_probability_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            ToolFailureConfig(failure_probability=-0.1)
        with pytest.raises(ValueError):
            ToolFailureConfig(failure_probability=1.1)

    @pytest.mark.parametrize(
        "failure_type",
        [ft for ft in ToolFailureType if ft != ToolFailureType.NONE],
    )
    def test_exported_config_works_with_injector(self, failure_type: ToolFailureType):
        # Regression: the old Pydantic export lacked malformed_payload /
        # retry_after_seconds / error_code and crashed here with AttributeError.
        cfg = ToolFailureConfig(
            failure_type=failure_type, failure_probability=1.0, timeout_seconds=0.001, seed=1
        )
        injector = ToolFailureInjector(cfg)
        with pytest.raises(_tool_failure_module.ToolCallError):
            injector.call(lambda x: x, "hi")
        assert injector.summary()["injected_failures"] == 1


class TestAgentResult:
    def test_successful_result(self):
        r = AgentResult(success=True, output="42", duration_ms=100.0)
        assert r.retries == 0
        assert r.error_message is None

    def test_duration_cannot_be_negative(self):
        with pytest.raises(ValidationError):
            AgentResult(success=True, duration_ms=-1.0)

    def test_from_invoke_dict(self):
        r = AgentResult.from_invoke({"output": "Paris", "latency_ms": 12.5, "framework": "x"})
        assert r.success and r.output == "Paris" and r.duration_ms == 12.5
        assert not AgentResult.from_invoke({"output": "", "latency_ms": 1.0}).success


class _StubAgent:
    framework = "stub"

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        return {"output": f"echo {input['input']}", "latency_ms": 1.0, "framework": self.framework}

    def reset(self) -> None:
        pass

    def get_injector_summary(self) -> dict[str, Any]:
        return {}


class TestBaseAgentProtocol:
    def test_stub_satisfies_protocol(self):
        assert isinstance(_StubAgent(), BaseAgent)

    def test_missing_reset_fails(self):
        class NoReset:
            framework = "x"

            def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
                return {}

            def get_injector_summary(self) -> dict[str, Any]:
                return {}

        assert not isinstance(NoReset(), BaseAgent)

    def test_langgraph_adapter_satisfies_protocol(self):
        from agentprobe.agents.langgraph_agent import LangGraphToolAgent

        agent = LangGraphToolAgent(llm=MagicMock())
        assert isinstance(agent, BaseAgent)
        agent.reset()  # no-op, must not raise
        assert agent.injector is None
        assert agent.get_injector_summary() == {}

    def test_langgraph_adapter_keeps_injector_across_reset(self):
        from agentprobe.agents.langgraph_agent import LangGraphToolAgent

        cfg = ToolFailureConfig(
            failure_type=ToolFailureType.API_ERROR, failure_probability=1.0, seed=3
        )
        agent = LangGraphToolAgent(llm=MagicMock(), failure_config=cfg)
        assert agent.injector is not None
        with pytest.raises(_tool_failure_module.ToolCallAPIError):
            agent.injector.call(lambda: None, tool_name="probe")
        agent.reset()
        assert agent.get_injector_summary()["injected_failures"] == 1  # condition-wide

    def test_langchain_adapter_satisfies_protocol(self):
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage

        from agentprobe.agents.langchain_agent import LangChainToolAgent

        llm = GenericFakeChatModel(messages=iter([AIMessage(content="hi")]))
        try:
            agent = LangChainToolAgent(llm=llm)
        except NotImplementedError:
            pytest.skip("fake chat model does not support bind_tools in this langchain version")
        assert isinstance(agent, BaseAgent)
        agent.reset()

    def test_autogen_adapter_satisfies_protocol(self):
        pytest.importorskip("autogen")
        from agentprobe.agents.autogen_agent import AutoGenToolAgent

        agent = AutoGenToolAgent(
            llm_config={"config_list": [{"model": "m", "api_base": "http://x", "api_key": "k"}]}
        )
        assert isinstance(agent, BaseAgent)
        agent.reset()

    def test_runner_accepts_protocol_agent(self):
        from agentprobe.benchmark import BenchmarkConfig, BenchmarkRunner

        results = BenchmarkRunner(agent=_StubAgent(), config=BenchmarkConfig(n_runs=2)).run()
        assert [r.success for r in results] == [True, True]
        assert "echo" in results[0].output


class TestPromptSets:
    def test_paper_prompts_are_the_runner_default(self):
        from agentprobe.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(agent=_StubAgent())
        assert runner._task_fn is paper_task_fn

    def test_paper_prompts_frozen(self):
        # These exact strings are what every published batch used; do not edit.
        assert PAPER_PROMPTS["tool_use"][0] == "What is the capital of France?"
        assert PAPER_PROMPTS["tool_use"][2] == "Calculate 15% of 847."
        assert all(len(v) == 5 for v in PAPER_PROMPTS.values())
        assert paper_task_fn(7, "tool_use")["input"] == PAPER_PROMPTS["tool_use"][2]

    @pytest.mark.parametrize("name,size", [("tool_use", 25), ("rag", 15), ("multi_agent", 15)])
    def test_corpus_task_fn_cycles_full_corpus(self, name: str, size: int):
        task = get_task(name)
        assert len(task) == size
        fn = corpus_task_fn(name)
        first, wrapped = fn(0, name), fn(size, name)
        assert first["query_id"] == wrapped["query_id"]
        assert first["run_id"] == 0 and wrapped["run_id"] == size
        assert {fn(i, name)["query_id"] for i in range(size)} == {
            q["query_id"] for q in task.get_inputs()
        }

    def test_corpus_inputs_are_gradable(self):
        fn = corpus_task_fn("rag")
        item = fn(0, "rag")
        assert "context" in item
        assert get_task("rag").grade(item["query_id"], "tool orchestration context latency")

    def test_unknown_task_rejected(self):
        with pytest.raises(ValueError):
            get_task("nope")


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
