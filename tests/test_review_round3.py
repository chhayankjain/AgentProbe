"""Regression tests for the third code review. Each test names its finding."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any

import pytest

from agentprobe.benchmark.metrics import MetricsCalculator
from agentprobe.benchmark.runner import BenchmarkConfig, BenchmarkRunner
from agentprobe.errors import ParallelToolCallsUnsupported
from agentprobe.fake_llm import ScriptedToolCallingModel
from agentprobe.injector.tool_failure import (
    ToolFailureConfig,
    ToolFailureInjector,
    ToolFailureType,
)
from agentprobe.llm import LLMSpec, autogen_llm_config, is_llm_backend_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "experiments"))

httpx = pytest.importorskip("httpx")


def _cfg(**kw: Any) -> BenchmarkConfig:
    base = dict(
        n_runs=1,
        recovery_attempts=1,
        backend_error_retries=2,
        backend_error_backoff_seconds=0.0,
        timeout_per_run_seconds=30.0,
    )
    base.update(kw)
    return BenchmarkConfig(**base)  # type: ignore[arg-type]


# --- 1. Backend errors are identified at the LLM-call boundary -----------------


class _FailingModel(ScriptedToolCallingModel):
    """Chat model whose call itself fails with a transport error."""

    def _generate(self, *a: Any, **k: Any) -> Any:
        raise httpx.ReadTimeout("provider read timeout")


def _tool_named(agent: Any, name: str) -> Any:
    return next(t for t in agent._tools if t.name == name)


def test_tool_transport_error_is_measured_as_failure():
    """A real tool raising httpx.ReadTimeout (no LLM involved) is not excluded."""
    from agentprobe.agents.langgraph_agent import LangGraphToolAgent

    agent = LangGraphToolAgent(llm=ScriptedToolCallingModel())

    def broken_service(expression: str) -> str:
        raise httpx.ReadTimeout("downstream service timed out")

    _tool_named(agent, "calculator").func = broken_service
    with pytest.raises(httpx.ReadTimeout) as info:
        agent.invoke({"input": "Calculate 15% of 847."})
    assert not is_llm_backend_error(info.value)

    r = BenchmarkRunner(agent=agent, config=_cfg()).run()[0]
    assert r.status == "completed" and not r.success
    assert r.failure_category != "infrastructure"
    m = MetricsCalculator().compute([r])
    assert m.n_executed == 1 and m.failure_rate == 1.0


@pytest.mark.parametrize("framework", ["langgraph", "langchain"])
def test_model_call_transport_error_is_backend_error(framework: str):
    if framework == "langgraph":
        from agentprobe.agents.langgraph_agent import LangGraphToolAgent as Agent
    else:
        from agentprobe.agents.langchain_agent import LangChainToolAgent as Agent
    agent = Agent(llm=_FailingModel())
    with pytest.raises(Exception) as info:
        agent.invoke({"input": "Calculate 15% of 847."})
    assert is_llm_backend_error(info.value)
    r = BenchmarkRunner(agent=agent, config=_cfg(backend_error_retries=0)).run()[0]
    assert r.status == "backend_error"


def test_autogen_shim_marks_model_call_errors(monkeypatch):
    pytest.importorskip("autogen")
    import openai
    from openai import error as oe

    from agentprobe.agents import autogen_agent as ag

    ag._patch_openai_compat()

    def failing(*a: Any, **k: Any) -> Any:
        raise oe.InvalidRequestError("bad", param=None)  # not transient: re-raised at once

    monkeypatch.setattr(ag, "_ORIGINAL_CREATE", failing)
    with pytest.raises(oe.InvalidRequestError) as info:
        openai.ChatCompletion.create(model="m", messages=[])
    assert getattr(info.value, "__agentprobe_llm_call__", False)


# --- 2. AutoGen must not silently drop parallel tool calls ---------------------


def _two_calls_response() -> dict:
    calls = [
        {
            "id": f"c{i}",
            "type": "function",
            "function": {"name": "calculator", "arguments": f'{{"expression": "{i}+1"}}'},
        }
        for i in (1, 2)
    ]
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": calls}}]}


def test_parallel_tool_calls_rejected_not_truncated(monkeypatch):
    pytest.importorskip("autogen")
    import autogen

    from agentprobe.agents import autogen_agent as ag

    ag._patch_openai_compat()
    monkeypatch.setattr(ag, "_ORIGINAL_CREATE", lambda *a, **k: _two_calls_response())
    with pytest.raises(ParallelToolCallsUnsupported) as info:
        autogen.ChatCompletion._get_response(
            {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}],
                "retry_wait_time": 0,
            },
            use_cache=False,
        )
    assert info.value.n_calls == 2
    assert not is_llm_backend_error(info.value)


def test_runner_excludes_harness_limitations():
    class Parallel:
        framework = "mock"
        calls = 0

        def invoke(self, input: dict) -> dict:
            Parallel.calls += 1
            raise ParallelToolCallsUnsupported(2, ["calculator", "calculator"])

    r = BenchmarkRunner(agent=Parallel(), config=_cfg()).run()[0]
    assert r.status == "harness_error" and r.failure_category == "harness"
    assert Parallel.calls == 1  # no recovery attempt
    m = MetricsCalculator().compute([r])
    assert m.n_executed == 0 and m.n_harness_errors == 1 and not m.condition_complete


def test_parallel_flag_sent_only_with_tools_and_never_leaks(monkeypatch):
    pytest.importorskip("autogen")
    import openai

    from agentprobe.agents import autogen_agent as ag

    ag._patch_openai_compat()
    seen: dict[str, Any] = {}

    def capture(*a: Any, **k: Any) -> dict:
        seen.clear()
        seen.update(k)
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(ag, "_ORIGINAL_CREATE", capture)
    fn = [{"name": "calculator", "parameters": {"type": "object", "properties": {}}}]
    openai.ChatCompletion.create(
        model="m", messages=[], functions=fn, agentprobe_parallel_tool_calls=False
    )
    assert seen["parallel_tool_calls"] is False and "agentprobe_parallel_tool_calls" not in seen
    openai.ChatCompletion.create(model="m", messages=[], agentprobe_parallel_tool_calls=False)
    assert "parallel_tool_calls" not in seen and "agentprobe_parallel_tool_calls" not in seen


def test_parallel_flag_only_for_supporting_providers(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert (
        autogen_llm_config(LLMSpec.from_provider("groq", model="m"))[
            "agentprobe_parallel_tool_calls"
        ]
        is False
    )
    assert "agentprobe_parallel_tool_calls" not in autogen_llm_config(
        LLMSpec.from_provider("ollama", model="m")
    )


# --- 3. Tool evidence is per trial -----------------------------------------------


def test_runner_records_per_trial_tool_evidence():
    from agentprobe.agents.langgraph_agent import LangGraphToolAgent

    agent = LangGraphToolAgent(llm=ScriptedToolCallingModel())
    results = BenchmarkRunner(agent=agent, config=_cfg(n_runs=3)).run()
    assert [r.trial_tool_calls for r in results] == [1, 1, 1]
    assert [r.trial_injections for r in results] == [0, 0, 0]


def test_recovery_tool_evidence_is_separate():
    from agentprobe.agents.langgraph_agent import LangGraphToolAgent

    cfg = ToolFailureConfig(failure_type=ToolFailureType.API_ERROR, failure_probability=1.0, seed=1)
    agent = LangGraphToolAgent(llm=ScriptedToolCallingModel(), failure_config=cfg)
    r = BenchmarkRunner(agent=agent, config=_cfg(failure_config=cfg)).run()[0]
    assert (r.trial_tool_calls, r.trial_injections) == (1, 1)
    assert (r.recovery_tool_calls, r.recovery_injections) == (1, 1)
    assert r.recovery_status == "failed"


def test_experiment_records_per_trial_and_condition_totals(monkeypatch, tmp_path):
    import run_experiment_v2 as v2

    from agentprobe.benchmark.grader import ResponseGrader

    monkeypatch.setattr(v2, "build_llm", lambda model, llm_spec=None: ScriptedToolCallingModel())
    outcome = v2.run_single_condition(
        framework="langgraph",
        task="tool_use",
        failure_type="none",
        n_runs=3,
        model="m",
        batch_id="b",
        batch_dir=tmp_path,
        grader=ResponseGrader(threshold=0.5),
    )
    rows = outcome.records
    assert [r.tool_call_count for r in rows] == [1, 1, 1]  # per trial
    assert [r.trial_tool_calls for r in rows] == [1, 1, 1]
    assert {r.injector_calls for r in rows} == {3}  # condition total, documented as such


# --- 4. Cohen's d with constant groups -------------------------------------------


def test_cohens_d_constant_groups():
    from agentprobe.benchmark.statistics import cohens_d

    d, interp = cohens_d([1, 1, 1], [100, 100, 100])
    assert math.isnan(d) and interp == "undefined"
    assert cohens_d([5, 5, 5], [5, 5, 5]) == (0.0, "negligible")


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_inputs_rejected(bad):
    from agentprobe.benchmark import statistics as st

    with pytest.raises(ValueError):
        st.cohens_d([1.0, bad, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        st.compare_distributions([1.0, bad], [1.0, 2.0])
    with pytest.raises(ValueError):
        st.compare_multiple_groups([[1.0, bad], [1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        st.cohens_h(bad, 0.5)


def test_comparison_report_with_constant_latencies():
    from agentprobe.benchmark.statistics import FrameworkData, build_comparison_report

    fws = [
        FrameworkData(name="a", successes=8, total=10, latencies=[1.0, 1.0, 1.0]),
        FrameworkData(name="b", successes=8, total=10, latencies=[100.0, 100.0, 100.0]),
        FrameworkData(name="c", successes=8, total=10, latencies=[1.0, 1.0, 1.0]),
    ]
    report = build_comparison_report(fws)
    ab = next(p for p in report.pairwise if (p.framework_a, p.framework_b) == ("a", "b"))
    assert ab.latency_effect is not None and ab.latency_effect[1] == "undefined"
    ac = next(p for p in report.pairwise if (p.framework_a, p.framework_b) == ("a", "c"))
    assert ac.latency_effect == (0.0, "negligible")
    identical = build_comparison_report(
        [FrameworkData(name=n, successes=1, total=2, latencies=[3.0, 3.0]) for n in "xyz"]
    )
    assert identical.omnibus_latency is not None
    assert identical.omnibus_latency.p_value == 1.0 and not identical.omnibus_latency.significant


# --- 5. Malformed payloads are reproducible from the seed ------------------------


def _payloads(seed: int, n: int = 6) -> list[str]:
    inj = ToolFailureInjector(
        ToolFailureConfig(
            failure_type=ToolFailureType.MALFORMED_OUTPUT, failure_probability=1.0, seed=seed
        )
    )
    out = []
    for _ in range(n):
        random.random()  # unrelated global RNG activity must not matter
        try:
            inj.call(lambda: "x")
        except Exception as e:
            out.append(str(e))
    return out


def test_malformed_payloads_reproducible_from_seed():
    random.seed(1)
    a = _payloads(42)
    random.seed(999)
    b = _payloads(42)
    assert a == b


def test_payload_stream_does_not_shift_injection_schedule():
    def schedule(ft: ToolFailureType) -> list[bool]:
        inj = ToolFailureInjector(
            ToolFailureConfig(
                failure_type=ft, failure_probability=0.5, seed=7, timeout_seconds=0.001
            )
        )
        for _ in range(20):
            try:
                inj.call(lambda: "x")
            except Exception:
                pass
        return [r.injected for r in inj.injection_log]

    assert schedule(ToolFailureType.MALFORMED_OUTPUT) == schedule(ToolFailureType.API_ERROR)


# --- 7. Zero drop fraction keeps every message ------------------------------------


def test_zero_drop_fraction_keeps_all_messages():
    from agentprobe.injector.context_failure import (
        ContextFailureConfig,
        ContextFailureInjector,
        ContextFailureType,
    )

    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    inj = ContextFailureInjector(
        ContextFailureConfig(failure_type=ContextFailureType.LOST_STATE, drop_fraction=0.0, seed=1)
    )
    assert inj.corrupt_messages(msgs) == msgs


# --- Protocol labels ---------------------------------------------------------------


def test_protocols_are_labelled_and_distinct():
    import run_benchmarks
    import run_experiment_v2

    assert run_experiment_v2.PROTOCOL.startswith("phase2-matched")
    assert run_benchmarks.PROTOCOL.startswith("phase1-discovery")
    assert "protocol" in run_experiment_v2.RESUME_COMPAT_KEYS
