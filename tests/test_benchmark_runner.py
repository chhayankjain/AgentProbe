"""Tests for BenchmarkRunner — multi-run experiment orchestration."""

from __future__ import annotations

import pytest

from agentprobe.benchmark.runner import BenchmarkConfig, BenchmarkRunner, RunResult

# Optional — only available after feature/injector-tool-failure is merged
try:
    from agentprobe.injector.tool_failure import ToolFailureConfig, ToolFailureType
    _INJECTOR_AVAILABLE = True
except ImportError:
    _INJECTOR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Mock agents
# ---------------------------------------------------------------------------


class AlwaysSuccessAgent:
    framework = "mock"

    def invoke(self, input: dict) -> dict:
        return {"output": "mock output", "latency_ms": 10.0}


class AlwaysFailAgent:
    framework = "mock"

    def invoke(self, input: dict) -> dict:
        raise RuntimeError("Mock agent failure")


class FlipFlopAgent:
    """Fails on even run_ids (input-driven, no shared mutable state)."""
    framework = "mock"

    def invoke(self, input: dict) -> dict:
        run_id = input.get("run_id", 0)
        if run_id % 2 == 1:
            raise RuntimeError("Odd run_id failure")
        return {"output": "ok", "latency_ms": 5.0}


class CustomOutputAgent:
    framework = "mock"

    def __init__(self, output: str = "custom output"):
        self._output = output

    def invoke(self, input: dict) -> dict:
        return {"output": self._output, "latency_ms": 25.0}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBenchmarkRunner:
    def _config(self, n_runs: int = 5, recovery: int = 0, **kwargs) -> BenchmarkConfig:
        return BenchmarkConfig(
            framework="mock",
            task="tool_use",
            n_runs=n_runs,
            recovery_attempts=recovery,
            **kwargs,
        )

    def test_all_success_all_runs_succeed(self):
        runner = BenchmarkRunner(AlwaysSuccessAgent(), self._config(n_runs=5))
        results = runner.run()
        assert len(results) == 5
        assert all(r.success for r in results)

    def test_all_failure_all_runs_fail(self):
        runner = BenchmarkRunner(AlwaysFailAgent(), self._config(n_runs=5))
        results = runner.run()
        assert len(results) == 5
        assert all(not r.success for r in results)

    def test_flip_flop_exactly_half_fail(self):
        runner = BenchmarkRunner(FlipFlopAgent(), self._config(n_runs=10))
        results = runner.run()
        successes = sum(1 for r in results if r.success)
        failures = sum(1 for r in results if not r.success)
        assert successes == 5
        assert failures == 5

    def test_correct_run_count(self):
        for n in [1, 10, 25]:
            runner = BenchmarkRunner(AlwaysSuccessAgent(), self._config(n_runs=n))
            results = runner.run()
            assert len(results) == n

    def test_run_result_framework_and_task(self):
        runner = BenchmarkRunner(AlwaysSuccessAgent(), self._config(n_runs=1))
        results = runner.run()
        assert results[0].framework == "mock"
        assert results[0].task == "tool_use"

    def test_run_result_run_id_sequential(self):
        runner = BenchmarkRunner(AlwaysSuccessAgent(), self._config(n_runs=5))
        results = runner.run()
        assert [r.run_id for r in results] == [0, 1, 2, 3, 4]

    def test_run_result_latency_non_negative(self):
        runner = BenchmarkRunner(AlwaysSuccessAgent(), self._config(n_runs=3))
        for r in runner.run():
            assert r.latency_ms >= 0.0

    def test_failure_result_has_exception_type(self):
        runner = BenchmarkRunner(AlwaysFailAgent(), self._config(n_runs=3))
        for r in runner.run():
            assert not r.success
            assert r.exception_type == "RuntimeError"
            assert "Mock agent failure" in r.exception_msg

    def test_failure_result_has_category(self):
        runner = BenchmarkRunner(AlwaysFailAgent(), self._config(n_runs=3))
        for r in runner.run():
            assert r.failure_category != ""

    def test_failure_type_label_none_baseline(self):
        runner = BenchmarkRunner(AlwaysSuccessAgent(), self._config(n_runs=1))
        results = runner.run()
        assert results[0].failure_type == "none"

    @pytest.mark.skipif(not _INJECTOR_AVAILABLE, reason="requires feature/injector-tool-failure")
    def test_failure_type_label_from_config(self):
        fc = ToolFailureConfig(
            failure_type=ToolFailureType.TIMEOUT,
            failure_probability=0.0,  # don't actually inject
        )
        config = self._config(n_runs=1, failure_config=fc)
        runner = BenchmarkRunner(AlwaysSuccessAgent(), config)
        results = runner.run()
        assert results[0].failure_type == "timeout"

    def test_to_dict_has_required_keys(self):
        runner = BenchmarkRunner(AlwaysSuccessAgent(), self._config(n_runs=1))
        r = runner.run()[0]
        d = r.to_dict()
        for key in ["run_id", "framework", "task", "success", "latency_ms", "failure_type", "status"]:
            assert key in d

    def test_custom_task_fn(self):
        inputs_seen = []

        def custom_task(run_id: int, task: str) -> dict:
            inputs_seen.append({"run_id": run_id, "task": task})
            return {"input": f"custom input {run_id}"}

        runner = BenchmarkRunner(
            AlwaysSuccessAgent(),
            self._config(n_runs=3),
            task_fn=custom_task,
        )
        runner.run()
        assert len(inputs_seen) == 3
        assert inputs_seen[0]["run_id"] == 0
        assert inputs_seen[2]["run_id"] == 2

    def test_save_results_creates_csv_and_json(self, tmp_path):
        save_path = str(tmp_path / "results" / "test_run")
        config = BenchmarkConfig(
            framework="mock",
            task="tool_use",
            n_runs=3,
            recovery_attempts=0,
            save_results_path=save_path,
        )
        runner = BenchmarkRunner(AlwaysSuccessAgent(), config)
        runner.run()

        import os
        assert os.path.exists(save_path + ".csv")
        assert os.path.exists(save_path + ".json")


class TestTimeoutIsolation:
    """Verify timeout handling, upstream error classification, and overlap prevention."""

    def test_hung_invocation_skips_recovery(self):
        """When an invocation hangs past the deadline, recovery is skipped
        (no second invocation on the same agent while the first is still alive).
        """
        call_count = {"n": 0}

        class SlowAgent:
            framework = "mock"

            def invoke(self, input: dict) -> dict:
                import time
                call_count["n"] += 1
                time.sleep(5)  # will exceed the 100ms deadline
                return {"output": "slow"}

        config = BenchmarkConfig(
            framework="mock",
            task="tool_use",
            n_runs=1,
            recovery_attempts=1,
            timeout_per_run_seconds=0.1,
        )
        runner = BenchmarkRunner(SlowAgent(), config)
        results = runner.run()

        assert len(results) == 1
        assert not results[0].success
        assert results[0].exception_type == "_HarnessTimeout"
        # Recovery must be skipped — only 1 invocation, not 2
        assert not results[0].recovered
        assert call_count["n"] == 1, (
            f"Expected 1 invocation (no recovery), got {call_count['n']}"
        )

    def test_upstream_timeout_preserves_type(self):
        """An agent raising TimeoutError should not be mislabeled as harness deadline."""

        class UpstreamTimeoutAgent:
            framework = "mock"

            def invoke(self, input: dict) -> dict:
                raise TimeoutError("upstream deadline exceeded")

        config = BenchmarkConfig(
            framework="mock",
            task="tool_use",
            n_runs=1,
            recovery_attempts=0,
            timeout_per_run_seconds=60.0,
        )
        runner = BenchmarkRunner(UpstreamTimeoutAgent(), config)
        results = runner.run()

        assert len(results) == 1
        r = results[0]
        assert not r.success
        assert r.exception_type == "TimeoutError"
        assert "upstream deadline" in r.exception_msg
        assert "_HarnessTimeout" not in r.exception_type

    def test_harness_timeout_returns_promptly(self):
        """The runner should return within a reasonable time after the deadline."""
        import time

        class HangAgent:
            framework = "mock"

            def invoke(self, input: dict) -> dict:
                time.sleep(60)
                return {"output": "never"}

        config = BenchmarkConfig(
            framework="mock",
            task="tool_use",
            n_runs=1,
            recovery_attempts=0,
            timeout_per_run_seconds=0.2,
        )
        runner = BenchmarkRunner(HangAgent(), config)
        start = time.perf_counter()
        results = runner.run()
        elapsed = time.perf_counter() - start

        assert len(results) == 1
        assert not results[0].success
        assert results[0].exception_type == "_HarnessTimeout"
        # Should return within 2s (generous margin for thread overhead)
        assert elapsed < 2.0, f"Timeout took {elapsed:.1f}s, expected < 2s"

    def test_injector_rng_advances_across_trials(self):
        """The injector's RNG must produce different values across trials,
        not repeat the same state (which would happen with fork-based isolation).
        """
        import random

        rng_values = []

        class RNGProbeAgent:
            framework = "mock"

            def __init__(self):
                self._rng = random.Random(42)

            def invoke(self, input: dict) -> dict:
                val = self._rng.random()
                rng_values.append(val)
                return {"output": str(val)}

        config = BenchmarkConfig(
            framework="mock",
            task="tool_use",
            n_runs=4,
            recovery_attempts=0,
        )
        runner = BenchmarkRunner(RNGProbeAgent(), config)
        runner.run()

        assert len(rng_values) == 4
        # All values must be distinct (RNG advancing, not resetting)
        assert len(set(rng_values)) == 4, (
            f"RNG produced duplicate values: {rng_values}"
        )

    def test_large_output_not_misclassified(self):
        """A large successful output must not be recorded as a timeout."""

        class LargeOutputAgent:
            framework = "mock"

            def invoke(self, input: dict) -> dict:
                return {"output": "x" * 100_000}

        config = BenchmarkConfig(
            framework="mock",
            task="tool_use",
            n_runs=1,
            recovery_attempts=0,
            timeout_per_run_seconds=10.0,
        )
        runner = BenchmarkRunner(LargeOutputAgent(), config)
        results = runner.run()

        assert len(results) == 1
        assert results[0].success, (
            f"Large output was misclassified: {results[0].exception_type}"
        )

    def test_timeout_aborts_remaining_trials(self):
        """After a harness timeout, remaining trials must be skipped — no new invocations."""
        call_count = {"n": 0}

        class SlowAgent:
            framework = "mock"

            def invoke(self, input: dict) -> dict:
                import time
                call_count["n"] += 1
                time.sleep(5)
                return {"output": "slow"}

        config = BenchmarkConfig(
            framework="mock",
            task="tool_use",
            n_runs=3,
            recovery_attempts=0,
            timeout_per_run_seconds=0.1,
        )
        runner = BenchmarkRunner(SlowAgent(), config)
        results = runner.run()

        assert len(results) == 3
        # Only the first trial should have been dispatched
        assert call_count["n"] == 1, (
            f"Expected 1 invocation, got {call_count['n']}"
        )
        # First trial: actual timeout, status=completed (it was attempted)
        assert results[0].exception_type == "_HarnessTimeout"
        assert not results[0].success
        assert results[0].status == "completed"
        # Remaining trials: skipped
        for r in results[1:]:
            assert not r.success
            assert r.status == "skipped"
            assert r.exception_type == "_HarnessTimeout"
            assert "Skipped" in r.exception_msg
            assert r.latency_ms == 0.0
            assert r.failure_category == "harness_abort"

    def test_skipped_trials_excluded_from_metrics(self):
        """MetricsCalculator must not count skipped trials in denominators."""
        from agentprobe.benchmark.metrics import MetricsCalculator

        results = [
            RunResult(
                run_id=0, framework="mock", task="tool_use",
                failure_type="none", success=False, latency_ms=100.0,
                status="completed", exception_type="_HarnessTimeout",
            ),
            RunResult(
                run_id=1, framework="mock", task="tool_use",
                failure_type="none", success=False, latency_ms=0.0,
                status="skipped",
            ),
            RunResult(
                run_id=2, framework="mock", task="tool_use",
                failure_type="none", success=False, latency_ms=0.0,
                status="skipped",
            ),
        ]

        calc = MetricsCalculator()
        metrics = calc.compute(results)

        # Only 1 executed trial, not 3
        assert metrics.n_executed == 1
        assert metrics.n_skipped == 2
        assert metrics.n_planned == 3
        assert not metrics.condition_complete
        # Failure rate from executed trials only
        assert metrics.n_runs == 1
        assert metrics.n_failures == 1
        assert metrics.failure_rate == 1.0
        # Latency from executed trials only (100ms, not averaged with 0s)
        assert metrics.mean_latency_ms == 100.0

    def test_trace_context_propagated_to_worker(self):
        """Agent invocation in the worker thread must inherit the benchmark span's trace."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

        class _CollectingExporter(SpanExporter):
            def __init__(self):
                self.spans = []

            def export(self, spans):
                self.spans.extend(spans)
                return SpanExportResult.SUCCESS

            def shutdown(self):
                pass

        exporter = _CollectingExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        old_provider = trace.get_tracer_provider()
        trace.set_tracer_provider(provider)

        observed_ctx = {}

        class TracingAgent:
            framework = "mock"

            def invoke(self, input: dict) -> dict:
                span = trace.get_current_span()
                ctx = span.get_span_context()
                observed_ctx["trace_id"] = ctx.trace_id
                observed_ctx["valid"] = ctx.is_valid
                return {"output": "traced"}

        try:
            config = BenchmarkConfig(
                framework="mock",
                task="tool_use",
                n_runs=1,
                recovery_attempts=0,
            )
            runner = BenchmarkRunner(TracingAgent(), config)
            results = runner.run()

            assert results[0].success

            benchmark_spans = [s for s in exporter.spans if s.name.startswith("benchmark.run.")]
            assert len(benchmark_spans) == 1

            # Agent observed a valid trace context belonging to the benchmark trace
            assert observed_ctx.get("valid", False), "Worker had no valid OTel context"
            assert observed_ctx["trace_id"] == benchmark_spans[0].context.trace_id, (
                "Worker trace ID does not match benchmark span trace ID"
            )
        finally:
            trace.set_tracer_provider(old_provider)
            provider.shutdown()

    def test_recovery_timeout_stops_retry_loop(self):
        """If a recovery attempt itself times out, further retries must stop."""
        call_count = {"n": 0}

        class FailThenHangAgent:
            framework = "mock"

            def invoke(self, input: dict) -> dict:
                import time
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise RuntimeError("initial failure")
                # Recovery attempt hangs
                time.sleep(5)
                return {"output": "never"}

        config = BenchmarkConfig(
            framework="mock",
            task="tool_use",
            n_runs=1,
            recovery_attempts=3,  # multiple retries configured
            timeout_per_run_seconds=0.1,
        )
        runner = BenchmarkRunner(FailThenHangAgent(), config)
        results = runner.run()

        assert len(results) == 1
        assert not results[0].success
        # Initial call + 1 recovery attempt that hung = 2 total
        # The remaining 2 recovery attempts must NOT have been dispatched
        assert call_count["n"] == 2, (
            f"Expected 2 invocations (initial + 1 hung recovery), got {call_count['n']}"
        )

    def test_non_hung_failure_allows_recovery(self):
        """When the invocation fails immediately (not hung), recovery should proceed."""
        call_count = {"n": 0}

        class FailThenSucceedAgent:
            framework = "mock"

            def invoke(self, input: dict) -> dict:
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise RuntimeError("first call fails")
                return {"output": "recovered"}

        config = BenchmarkConfig(
            framework="mock",
            task="tool_use",
            n_runs=1,
            recovery_attempts=1,
        )
        runner = BenchmarkRunner(FailThenSucceedAgent(), config)
        results = runner.run()

        assert len(results) == 1
        assert not results[0].success  # initial invocation failed
        assert results[0].recovered  # recovery succeeded
        assert call_count["n"] == 2  # initial + recovery
