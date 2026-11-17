"""Regression coverage for incomplete experiment reporting and saved provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentprobe.benchmark.grader import ResponseGrader
from agentprobe.benchmark.metrics import MetricsCalculator
from agentprobe.benchmark.runner import BenchmarkConfig, BenchmarkRunner, RunResult
from agentprobe.errors import ParallelToolCallsUnsupported
from agentprobe.llm import LLMSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))


def _result(**overrides):
    args = dict(
        run_id=0,
        framework="autogen",
        task="tool_use",
        failure_type="none",
        success=False,
        latency_ms=1.0,
    )
    args.update(overrides)
    return RunResult(**args)


def test_legacy_matrix_stops_at_an_incomplete_condition(monkeypatch):
    import run_benchmarks as script

    class Unsupported:
        def invoke(self, input):
            raise ParallelToolCallsUnsupported(2, ["calculator", "web_search"])

    built = []

    def build(*args, **kwargs):
        built.append(1)
        return Unsupported()

    monkeypatch.setattr(script, "build_agent", build)
    metrics = script.run_matrix(
        ["autogen"], ["tool_use"], ["none", "api_error"], 1, LLMSpec(), False
    )
    assert len(built) == len(metrics) == 1
    assert metrics[0]["n_harness_errors"] == 1
    assert not metrics[0]["condition_complete"]


@pytest.mark.parametrize("complete", [False, True])
def test_cli_preserves_results_and_signals_incomplete_batch(monkeypatch, tmp_path, complete):
    import run_benchmarks as script

    metrics = (
        MetricsCalculator()
        .compute([_result(status="completed" if complete else "harness_error", success=complete)])
        .to_dict()
    )
    monkeypatch.setattr(script, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(script, "run_matrix", lambda **kwargs: [metrics])
    monkeypatch.setattr(
        script, "capture_metadata", lambda **kwargs: SimpleNamespace(to_dict=lambda: {})
    )
    monkeypatch.setattr(
        sys, "argv", ["run_benchmarks", "--baseline-only", "--save-results", "--n-runs", "1"]
    )
    if complete:
        script.main()
    else:
        with pytest.raises(SystemExit) as info:
            script.main()
        assert info.value.code == 1
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved[0]["condition_complete"] is complete


def test_summary_reports_unassessed_recovery_denominator():
    runner = BenchmarkRunner(object(), BenchmarkConfig(n_runs=2))
    summary = runner._summary_line(
        [
            _result(recovered=True, recovery_status="recovered"),
            _result(run_id=1, recovery_status="unassessed"),
        ]
    )
    assert "recovered=1/1 assessable" in summary
    assert "1 recoveries unassessed" in summary
    empty = runner._summary_line([_result(status="harness_error")])
    assert "failure_rate=N/A" in empty


def test_last_trial_hung_worker_marks_full_sized_condition_incomplete(monkeypatch):
    import run_benchmarks as script

    class Runner:
        _invocation_hung = True

        def __init__(self, **kwargs):
            pass

        def run(self):
            return [_result(exception_type="_HarnessTimeout")]

    monkeypatch.setattr(script, "BenchmarkRunner", Runner)
    monkeypatch.setattr(script, "build_agent", lambda *args, **kwargs: object())
    metrics = script.run_matrix(["autogen"], ["tool_use"], ["none"], 1, LLMSpec(), False)
    assert metrics[0]["n_executed"] == metrics[0]["n_planned"] == 1
    assert not metrics[0]["condition_complete"]


def test_v2_persists_backend_retry_provenance(monkeypatch, tmp_path):
    import run_experiment_v2 as script

    result = _result(success=True, backend_retries=2, output="Paris")

    class Runner:
        _invocation_hung = False

        def __init__(self, **kwargs):
            pass

        def run(self):
            return [result]

    monkeypatch.setattr(script, "BenchmarkRunner", Runner)
    monkeypatch.setattr(script, "build_agent", lambda *args, **kwargs: object())
    outcome = script.run_single_condition(
        "autogen", "tool_use", "none", 1, "fake", "batch", tmp_path, ResponseGrader()
    )
    assert outcome.records[0].backend_retries == 2
    saved = json.loads((tmp_path / "autogen_tool_use_none.json").read_text())
    assert saved[0]["backend_retries"] == 2
    assert script.record_from_dict(saved[0]).backend_retries == 2
    del saved[0]["backend_retries"]
    assert script.record_from_dict(saved[0]).backend_retries == 0
