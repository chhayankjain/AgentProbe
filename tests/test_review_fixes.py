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
from agentprobe.llm import (
    LLMSpec,
    autogen_effective_settings,
    autogen_llm_config,
    mark_llm_call_error,
)

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
            raise mark_llm_call_error(FakeRateLimitError("429 from provider"))
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


def test_autogen_effective_settings_reported():
    s = autogen_effective_settings(
        LLMSpec.from_provider("gemini", model="m", requests_per_minute=5)
    )
    assert s["requests_per_minute"] == 5 and s["stop_on_final_answer"] is True
    assert s["max_retry_period_seconds"] == (6 + 2) * 10


def test_agent_sets_its_own_pyautogen_retry_window():
    pytest.importorskip("autogen")
    from agentprobe.agents.autogen_agent import AutoGenToolAgent

    cfg = {"config_list": [{"model": "m", "api_base": "http://x", "api_key": "k"}]}
    agent = AutoGenToolAgent(llm_config=cfg, max_retries=2, requests_per_minute=30)
    assert agent._llm_config["max_retry_period"] == 40  # (2 + 2) * 10, per agent
    assert cfg.get("max_retry_period") is None  # caller's dict untouched
    eff = agent.effective_settings()
    assert eff["max_retries"] == 2 and eff["requests_per_minute"] == 30


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
    assert ag._state().cache == {}


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


# --- Round 2: pyautogen retries 503/connection/APIError forever --------------


@pytest.mark.parametrize("exc_name", ["ServiceUnavailableError", "APIConnectionError", "APIError"])
@pytest.mark.parametrize("max_retries", [0, 2])
def test_pyautogen_retry_loop_is_bounded(monkeypatch, exc_name, max_retries):
    """Drive pyautogen's real _get_response loop with real openai<1 exceptions."""
    autogen = pytest.importorskip("autogen")
    from openai import error as oe

    from agentprobe.agents import autogen_agent as ag
    from agentprobe.llm import BackendRetriesExhausted, is_llm_backend_error

    ag._patch_openai_compat()
    calls = {"n": 0}

    def failing(*a, **k):
        calls["n"] += 1
        if calls["n"] > 50:
            raise AssertionError("retry loop not bounded")
        raise getattr(oe, exc_name)("simulated provider failure")

    monkeypatch.setattr(ag, "_ORIGINAL_CREATE", failing)
    try:
        ag.configure_compat(max_retries=max_retries)
        with pytest.raises(BackendRetriesExhausted) as info:
            autogen.ChatCompletion._get_response(
                {
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                    "retry_wait_time": 0,
                },
                use_cache=False,
            )
        assert calls["n"] == max_retries + 1
        assert isinstance(info.value.__cause__, getattr(oe, exc_name))
        assert is_llm_backend_error(info.value)
    finally:
        ag.configure_compat()


def test_transient_error_then_success_resets_counter(monkeypatch):
    autogen = pytest.importorskip("autogen")
    from openai import error as oe

    from agentprobe.agents import autogen_agent as ag

    ag._patch_openai_compat()
    seq = iter([oe.ServiceUnavailableError("503"), {"choices": [{"message": {"content": "ok"}}]}])

    def flaky(*a, **k):
        item = next(seq)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(ag, "_ORIGINAL_CREATE", flaky)
    try:
        ag.configure_compat(max_retries=1)
        resp = autogen.ChatCompletion._get_response(
            {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}],
                "retry_wait_time": 0,
            },
            use_cache=False,
        )
        assert resp["choices"][0]["message"]["content"] == "ok"
        assert ag._state().consecutive == 0
    finally:
        ag.configure_compat()


def test_content_filter_and_invalid_request_are_not_counted():
    pytest.importorskip("autogen")
    from openai import error as oe

    from agentprobe.agents import autogen_agent as ag

    assert not ag._is_transient_provider_error(oe.InvalidRequestError("bad", param=None))
    cf = oe.APIError("filtered", json_body={"error": {"code": "content_filter"}})
    assert not ag._is_transient_provider_error(cf)
    assert ag._is_transient_provider_error(oe.ServiceUnavailableError("503"))


def test_runner_treats_exhausted_retries_as_backend_error():
    from agentprobe.llm import BackendRetriesExhausted

    class Exhausted:
        framework = "mock"

        def invoke(self, input):
            raise BackendRetriesExhausted("503 persisted")

    r = BenchmarkRunner(agent=Exhausted(), config=_cfg(backend_error_retries=0)).run()[0]
    assert r.status == "backend_error" and r.failure_category == "infrastructure"


# --- Round 2, item 2: resume compares by key presence, including None ---------


def test_resume_rejects_change_from_explicit_none():
    import run_experiment_v2 as r

    orig = {"model": "m", "llm_provider": "groq", "llm_requests_per_minute": None}
    cur = {"model": "m", "llm_provider": "groq", "llm_requests_per_minute": 10}
    problems = r.resume_mismatches(orig, cur)
    assert any("llm_requests_per_minute" in p for p in problems)


def test_resume_accepts_identical_config_including_none():
    import run_experiment_v2 as r

    meta = {k: None for k in r.RESUME_COMPAT_KEYS}
    meta.update(model="m", seed=42)
    assert r.resume_mismatches(dict(meta), dict(meta)) == []


def test_resume_of_legacy_batch_checks_documented_defaults():
    import run_experiment_v2 as r

    legacy = {
        "model": "llama3.1",
        "seed": 42,
        "n_runs_per_condition": 30,
        "injection_probability": 0.2,
    }  # pre-provider batch: no llm_* keys
    same = dict(
        legacy,
        protocol=r.PROTOCOL,
        llm_provider="ollama",
        llm_base_url="http://localhost:11434/v1",
        llm_requests_per_minute=None,
        llm_temperature=0.0,
        llm_timeout_seconds=120.0,
        llm_max_retries=6,
    )
    assert r.resume_mismatches(legacy, same) == []
    throttled = dict(same, llm_requests_per_minute=10)
    assert any("pre-provider" in p for p in r.resume_mismatches(legacy, throttled))
    hosted = dict(same, llm_provider="groq")
    assert any("llm_provider" in p for p in r.resume_mismatches(legacy, hosted))


# --- Round 2, item 3: throttle never leaks between differently configured agents


def _limiter_of_thread():
    from agentprobe.agents import autogen_agent as ag

    return ag._state().limiter


def test_unthrottled_configuration_clears_previous_limiter():
    pytest.importorskip("autogen")
    from agentprobe.agents import autogen_agent as ag

    ag.configure_compat(requests_per_minute=30)
    assert _limiter_of_thread() is not None
    ag.configure_compat(requests_per_minute=None)
    assert _limiter_of_thread() is None


def test_agents_activate_their_own_settings():
    pytest.importorskip("autogen")
    from agentprobe.agents import autogen_agent as ag
    from agentprobe.agents.autogen_agent import AutoGenToolAgent

    cfg = {"config_list": [{"model": "m", "api_base": "http://x", "api_key": "k"}]}
    throttled = AutoGenToolAgent(llm_config=cfg, requests_per_minute=30, max_retries=1)
    free = AutoGenToolAgent(llm_config=cfg)  # created after: must not inherit or clobber

    throttled.reset()
    assert _limiter_of_thread() is throttled._limiter and ag._state().max_retries == 1
    free.reset()
    assert _limiter_of_thread() is None and ag._state().max_retries == 6
    throttled.reset()
    assert _limiter_of_thread() is throttled._limiter


def test_concurrent_conversations_have_isolated_state():
    """Two threads, interleaved: each keeps its own cache, limiter and retry limit."""
    pytest.importorskip("autogen")
    import threading

    from agentprobe.agents import autogen_agent as ag

    barrier = threading.Barrier(2)
    seen: dict[str, tuple] = {}

    def worker(name: str, rpm, retries: int, call_id: str) -> None:
        ag.configure_compat(requests_per_minute=rpm, max_retries=retries)
        ag._remember_tool_call(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "calculator", "arguments": "{}"},
                "extra_content": {"sig": call_id},
            }
        )
        barrier.wait()  # both threads have written before either reads
        legacy = [
            {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "calculator", "arguments": "{}"},
            }
        ]
        restored = ag._restore_tool_calls(legacy)[0]["tool_calls"][0]["id"]
        seen[name] = (restored, ag._state().limiter is not None, ag._state().max_retries)

    a = threading.Thread(target=worker, args=("a", 30, 1, "call-a"))
    b = threading.Thread(target=worker, args=("b", None, 4, "call-b"))
    a.start()
    b.start()
    a.join()
    b.join()
    assert seen["a"] == ("call-a", True, 1)
    assert seen["b"] == ("call-b", False, 4)
