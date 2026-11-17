"""Regression tests for the code-review findings on the provider/AutoGen work.

Each test names the finding it guards.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agentprobe.benchmark.metrics import MetricsCalculator
from agentprobe.benchmark.runner import BenchmarkConfig, BenchmarkRunner, RunResult
from agentprobe.injector.tool_failure import ToolCallRateLimited
from agentprobe.llm import LLMSpec, autogen_effective_settings, autogen_llm_config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

FakeRateLimitError = type("RateLimitError", (Exception,), {"__module__": "openai"})


# --- Finding 1: AutoGen conditions must not construct a LangChain model ---------


def test_autogen_condition_never_builds_langchain_model(monkeypatch, tmp_path):
    import run_experiment_v2 as r

    class Reached(Exception):
        pass

    def no_llm(*a, **k):
        raise AssertionError("build_llm must not be called for autogen")

    def fake_agent(framework, llm, *a, **k):
        assert framework == "autogen" and llm is None
        raise Reached

    monkeypatch.setattr(r, "build_llm", no_llm)
    monkeypatch.setattr(r, "build_agent", fake_agent)
    with pytest.raises(Reached):
        r.run_single_condition(
            framework="autogen",
            task="tool_use",
            failure_type="none",
            n_runs=1,
            model="m",
            batch_id="b",
            batch_dir=tmp_path,
            grader=None,
        )


# --- Finding 2: backend errors during recovery -------------------------------


class InjectedThenBackend:
    """First call: injected tool fault. Recovery call(s): provider 429."""

    framework = "mock"

    def __init__(self, backend_failures: int) -> None:
        self.calls = 0
        self.backend_failures = backend_failures

    def invoke(self, input: dict) -> dict:
        self.calls += 1
        if self.calls == 1:
            raise ToolCallRateLimited()
        if self.calls - 1 <= self.backend_failures:
            raise FakeRateLimitError("429 from provider")
        return {"output": "ok"}


def _cfg(**kw):
    base = dict(
        n_runs=1, recovery_attempts=1, backend_error_retries=2, backend_error_backoff_seconds=0.0
    )
    base.update(kw)
    return BenchmarkConfig(**base)


def test_persistent_backend_error_during_recovery_is_unassessed():
    agent = InjectedThenBackend(backend_failures=99)
    r = BenchmarkRunner(agent=agent, config=_cfg()).run()[0]
    assert r.status == "completed" and not r.success
    assert r.recovered is False
    assert r.recovery_status == "unassessed"
    assert r.backend_retries == 2
    assert agent.calls == 1 + 3  # initial + recovery + 2 backend retries


def test_transient_backend_error_during_recovery_is_retried_and_recovers():
    agent = InjectedThenBackend(backend_failures=1)
    r = BenchmarkRunner(agent=agent, config=_cfg()).run()[0]
    assert r.recovered is True and r.recovery_status == "recovered"
    assert r.backend_retries == 1


def test_agent_failure_during_recovery_is_failed():
    class AlwaysInjected:
        framework = "mock"

        def invoke(self, input):
            raise ToolCallRateLimited()

    r = BenchmarkRunner(agent=AlwaysInjected(), config=_cfg()).run()[0]
    assert r.recovery_status == "failed" and r.backend_retries == 0


def test_unassessed_recovery_leaves_recovery_denominator():
    def rr(i, status, recovered=False):
        return RunResult(
            run_id=i,
            framework="f",
            task="t",
            failure_type="x",
            success=False,
            latency_ms=1.0,
            recovered=recovered,
            recovery_status=status,
        )

    m = MetricsCalculator().compute(
        [rr(0, "recovered", True), rr(1, "failed"), rr(2, "unassessed")]
    )
    assert m.failure_rate == 1.0  # still a failure of the initial run
    assert m.recovery_rate == 0.5  # 1 of 2 assessable
    assert m.n_recovery_unassessed == 1


# --- Finding 3: subset resume must not shrink batch-wide outputs ---------------


def test_load_completed_records_covers_all_checkpointed_conditions(tmp_path):
    import run_experiment_v2 as r

    state = r.ExperimentState()
    for ft in ("none", "timeout"):
        state.mark_done("langgraph", "tool_use", ft)
        rec = r.EnhancedRunRecord(
            run_id=0,
            framework="langgraph",
            task="tool_use",
            failure_type=ft,
            batch_id="b",
            status="completed",
            success=True,
            latency_ms=1.0,
            output="x",
        )
        (tmp_path / f"langgraph_tool_use_{ft}.json").write_text(json.dumps([rec.to_dict()]))
    state.save(tmp_path / "checkpoint.json")

    loaded = r.load_completed_records(
        tmp_path, r.ExperimentState.load(tmp_path / "checkpoint.json")
    )
    assert {x.failure_type for x in loaded} == {"none", "timeout"}


def test_record_from_dict_tolerates_old_and_unknown_fields():
    import run_experiment_v2 as r

    d = {
        "run_id": 0,
        "framework": "f",
        "task": "t",
        "failure_type": "x",
        "batch_id": "b",
        "status": "completed",
        "success": True,
        "latency_ms": 1.0,
        "output": "",
        "some_future_field": 1,
    }
    rec = r.record_from_dict(d)
    assert rec.recovery_status == "n/a"


# --- Finding 4: AutoGen effective settings ------------------------------------


def test_autogen_config_carries_temperature():
    cfg = autogen_llm_config(LLMSpec.from_provider("ollama", model="m", temperature=0.7))
    assert cfg["temperature"] == 0.7


def test_autogen_throttle_and_retry_window_are_applied():
    pytest.importorskip("autogen")
    from autogen import Completion

    from agentprobe.agents import autogen_agent as ag

    old = Completion.max_retry_period
    try:
        eff = ag.configure_compat(requests_per_minute=30, max_retries=2)
        assert ag._RATE_LIMITER is not None
        assert Completion.max_retry_period == 2 * Completion.retry_wait_time
        assert eff["requests_per_minute"] == 30 and eff["max_retries"] == 2
    finally:
        ag._RATE_LIMITER = None
        Completion.max_retry_period = old

    s = autogen_effective_settings(
        LLMSpec.from_provider("gemini", model="m", requests_per_minute=5)
    )
    assert s["requests_per_minute"] == 5 and s["stop_on_final_answer"] is True


# --- Finding 5: repeated identical tool calls keep their own identity --------


def test_repeated_identical_calls_restore_in_order():
    from agentprobe.agents import autogen_agent as ag

    ag._clear_tool_call_cache()
    args = '{"expression": "2 + 2"}'
    first = {
        "id": "c1",
        "type": "function",
        "function": {"name": "calculator", "arguments": args},
        "extra_content": {"google": {"thought_signature": "sig1"}},
    }
    second = dict(first, id="c2", extra_content={"google": {"thought_signature": "sig2"}})
    ag._remember_tool_call(first)
    ag._remember_tool_call(second)
    legacy = {
        "role": "assistant",
        "content": None,
        "function_call": {"name": "calculator", "arguments": args},
    }
    out = ag._restore_tool_calls(
        [legacy, {"role": "function", "name": "calculator", "content": "4"}, dict(legacy)]
    )
    assert out[0]["tool_calls"][0]["id"] == "c1"
    assert out[2]["tool_calls"][0]["id"] == "c2"


def test_cache_cleared_between_conversations():
    pytest.importorskip("autogen")
    from agentprobe.agents import autogen_agent as ag
    from agentprobe.agents.autogen_agent import AutoGenToolAgent

    ag._remember_tool_call(
        {"id": "x", "function": {"name": "f", "arguments": "{}"}, "extra_content": {"g": 1}}
    )
    agent = AutoGenToolAgent(
        llm_config={"config_list": [{"model": "m", "api_base": "http://x", "api_key": "k"}]}
    )
    agent.reset()
    assert ag._TOOL_CALL_CACHE == {}


# --- Finding 6: repeated tracer construction keeps library spans ------------


def test_second_tracer_receives_library_spans():
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from agentprobe.observer.tracer import AgentProbeTracer, get_tracer

    t1 = AgentProbeTracer(service_name="one", console=False)
    t2 = AgentProbeTracer(service_name="two", console=False)
    assert t1._provider is t2._provider
    exporter = InMemorySpanExporter()
    t2.add_span_processor(SimpleSpanProcessor(exporter))
    with get_tracer("agentprobe.some.module").start_as_current_span("library.span"):
        pass
    with t2.span("explicit.span"):
        pass
    t1.shutdown()  # must not silence the shared provider
    with get_tracer("agentprobe.other").start_as_current_span("after.shutdown"):
        pass
    names = {s.name for s in exporter.get_finished_spans()}
    assert {"library.span", "explicit.span", "after.shutdown"} <= names


# --- Demo: task matches the scripted model, answers are graded ---------------


def test_demo_grades_answers_against_matching_task():
    from agentprobe.demo import run_demo

    base = run_demo(probability=0.0, n_runs=4, quiet=True)
    assert base["n_correct"] == 4
    assert all(r.output and "127.05" in r.output for r in base["results"])
    faulty = run_demo(fault="api_error", probability=1.0, n_runs=3, quiet=True)
    assert faulty["n_correct"] == 0  # every call faults, no answer
