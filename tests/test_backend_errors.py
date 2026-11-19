"""LLM backend errors (429/5xx/connection) are infrastructure events, not agent failures.

They must never be counted alongside injected tool faults: the runner retries the
trial in place, then records ``status="backend_error"``, and the metrics exclude it.
"""

from __future__ import annotations

import pytest

from agentprobe.benchmark.metrics import MetricsCalculator
from agentprobe.benchmark.runner import BenchmarkConfig, BenchmarkRunner, RunResult
from agentprobe.injector.tool_failure import ToolCallRateLimited, ToolCallTimeout
from agentprobe.llm import mark_llm_call_error
from agentprobe.observer.classifier import FailureCategory, FailureClassifier

# A stand-in for openai.RateLimitError without importing the SDK.
FakeRateLimitError = type("RateLimitError", (Exception,), {"__module__": "openai"})


class FlakyBackendAgent:
    """Raises a provider error for the first *n_failures* invocations, then succeeds."""

    framework = "mock"

    def __init__(self, n_failures: int) -> None:
        self.n_failures = n_failures
        self.invocations = 0

    def invoke(self, input: dict) -> dict:
        self.invocations += 1
        if self.invocations <= self.n_failures:
            raise mark_llm_call_error(FakeRateLimitError("429 from provider"))
        return {"output": "ok", "latency_ms": 1.0}


class InjectedFaultAgent:
    framework = "mock"

    def __init__(self, fault_type: type[Exception]) -> None:
        self.fault_type = fault_type
        self.invocations = 0

    def invoke(self, input: dict) -> dict:
        self.invocations += 1
        raise self.fault_type()


def _cfg(**kw: object) -> BenchmarkConfig:
    base = dict(n_runs=1, recovery_attempts=1, backend_error_backoff_seconds=0.0)
    base.update(kw)
    return BenchmarkConfig(**base)  # type: ignore[arg-type]


class TestRunnerBackendErrors:
    def test_transient_backend_error_is_retried_in_place(self):
        agent = FlakyBackendAgent(n_failures=2)
        results = BenchmarkRunner(agent=agent, config=_cfg(backend_error_retries=2)).run()
        assert len(results) == 1
        r = results[0]
        assert r.status == "completed" and r.success
        assert r.backend_retries == 2
        assert agent.invocations == 3  # 2 failed dispatches + 1 success, no recovery call

    def test_persistent_backend_error_is_recorded_not_counted_as_failure(self):
        agent = FlakyBackendAgent(n_failures=99)
        results = BenchmarkRunner(agent=agent, config=_cfg(backend_error_retries=1)).run()
        r = results[0]
        assert r.status == "backend_error"
        assert r.failure_category == "infrastructure"
        assert r.exception_type == "RateLimitError"
        assert r.recovered is False
        assert r.backend_retries == 1
        # Dispatches: initial + 1 retry; recovery is never attempted for backend errors.
        assert agent.invocations == 2

    @pytest.mark.parametrize("fault_type", [ToolCallRateLimited, ToolCallTimeout])
    def test_injected_tool_fault_still_counts_as_agent_failure(self, fault_type):
        agent = InjectedFaultAgent(fault_type)
        results = BenchmarkRunner(agent=agent, config=_cfg(backend_error_retries=5)).run()
        r = results[0]
        assert r.status == "completed" and not r.success
        assert r.failure_category == "tool_call"
        assert r.exception_type == fault_type.__name__
        assert r.backend_retries == 0
        assert agent.invocations == 2  # initial + one recovery attempt, no backend retries

    def test_to_dict_includes_backend_retries(self):
        r = RunResult(
            run_id=0,
            framework="f",
            task="t",
            failure_type="none",
            success=True,
            latency_ms=1.0,
            backend_retries=3,
        )
        assert r.to_dict()["backend_retries"] == 3


class TestMetricsExcludeBackendErrors:
    def test_backend_errors_excluded_from_every_rate(self):
        runs = [
            RunResult(
                run_id=0,
                framework="f",
                task="t",
                failure_type="none",
                success=True,
                latency_ms=10.0,
            ),
            RunResult(
                run_id=1,
                framework="f",
                task="t",
                failure_type="none",
                success=False,
                latency_ms=20.0,
                failure_category="tool_call",
            ),
            RunResult(
                run_id=2,
                framework="f",
                task="t",
                failure_type="none",
                success=False,
                latency_ms=999.0,
                status="backend_error",
                failure_category="infrastructure",
            ),
            RunResult(
                run_id=3,
                framework="f",
                task="t",
                failure_type="none",
                success=False,
                latency_ms=0.0,
                status="skipped",
            ),
        ]
        m = MetricsCalculator().compute(runs)
        assert m.n_planned == 4
        assert m.n_executed == 2
        assert m.n_backend_errors == 1
        assert m.n_skipped == 1
        assert m.failure_rate == 0.5  # 1 of 2 executed, the 429 is not in the denominator
        assert m.max_latency_ms == 20.0  # backend-error latency excluded
        assert not m.condition_complete
        assert m.to_dict()["n_backend_errors"] == 1

    def test_dict_records_from_json_supported(self):
        runs = [
            {"status": "completed", "success": True, "latency_ms": 1.0},
            {"status": "backend_error", "success": False, "latency_ms": 500.0},
        ]
        m = MetricsCalculator().compute(runs, framework="f", task="t", failure_type="none")
        assert m.n_executed == 1 and m.n_backend_errors == 1 and m.failure_rate == 0.0


class TestClassifierInfrastructure:
    @pytest.mark.parametrize(
        "name",
        [
            "RateLimitError",
            "APITimeoutError",
            "APIConnectionError",
            "InternalServerError",
            "ConnectError",
        ],
    )
    def test_provider_errors_are_infrastructure_not_tool_call(self, name: str):
        exc = type(name, (Exception,), {"__module__": "openai"})("x")
        event = FailureClassifier().classify_exception(exc)  # name-based label only
        assert event.category == FailureCategory.INFRASTRUCTURE

    def test_injected_rate_limit_is_still_tool_call(self):
        event = FailureClassifier().classify_exception(ToolCallRateLimited())
        assert event.category == FailureCategory.TOOL_CALL
