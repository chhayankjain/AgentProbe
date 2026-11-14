"""Tests for ExperimentState checkpoint validation and resume safety."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from run_experiment_v2 import ExperimentState


def _rec(run_id: int, fw: str, task: str, ft: str) -> dict:
    return {
        "status": "completed",
        "run_id": run_id,
        "framework": fw,
        "task": task,
        "failure_type": ft,
    }


class TestExperimentStateValidation:
    def _make(
        self, tmp: Path, conditions: dict[tuple[str, str, str], list[dict] | int]
    ) -> ExperimentState:
        """Build checkpoint. Keys are (fw, task, ft) tuples.
        Value -1 = missing file, int >= 0 = auto-generate, list = exact records.
        """
        state = ExperimentState()
        for (fw, task, ft), value in conditions.items():
            state.mark_done(fw, task, ft)
            key = f"{fw}_{task}_{ft}"
            if isinstance(value, list):
                (tmp / f"{key}.json").write_text(json.dumps(value))
            elif value >= 0:
                records = [_rec(i, fw, task, ft) for i in range(value)]
                (tmp / f"{key}.json").write_text(json.dumps(records))
        state.save(tmp / "checkpoint.json")
        return ExperimentState.load(tmp / "checkpoint.json")

    # --- Basic cases ---

    def test_valid_checkpoint_passes(self, tmp_path):
        state = self._make(
            tmp_path,
            {
                ("langgraph", "tool_use", "baseline"): 30,
                ("langchain", "tool_use", "timeout"): 30,
            },
        )
        assert state.validate_checkpoint(tmp_path, n_runs=30) == []

    def test_missing_file_detected(self, tmp_path):
        state = self._make(
            tmp_path,
            {
                ("langgraph", "tool_use", "baseline"): -1,
            },
        )
        errors = state.validate_checkpoint(tmp_path, n_runs=30)
        assert len(errors) == 1
        assert "missing" in errors[0].lower()

    def test_too_few_records(self, tmp_path):
        state = self._make(
            tmp_path,
            {
                ("langgraph", "tool_use", "baseline"): 15,
            },
        )
        errors = state.validate_checkpoint(tmp_path, n_runs=30)
        assert len(errors) == 1
        assert "15" in errors[0]

    def test_too_many_records(self, tmp_path):
        state = self._make(
            tmp_path,
            {
                ("langgraph", "tool_use", "baseline"): 31,
            },
        )
        errors = state.validate_checkpoint(tmp_path, n_runs=30)
        assert len(errors) == 1
        assert "31" in errors[0]

    def test_empty_checkpoint_passes(self, tmp_path):
        state = ExperimentState()
        assert state.validate_checkpoint(tmp_path, n_runs=30) == []

    # --- Duplicate / wrong run IDs ---

    def test_duplicate_run_ids(self, tmp_path):
        dupes = [_rec(0, "langgraph", "tool_use", "baseline")] * 30
        state = self._make(
            tmp_path,
            {
                ("langgraph", "tool_use", "baseline"): dupes,
            },
        )
        errors = state.validate_checkpoint(tmp_path, n_runs=30)
        assert any("duplicate" in e.lower() for e in errors)

    def test_wrong_run_id_range(self, tmp_path):
        """IDs 100-129 should fail when 0-29 are expected."""
        wrong_ids = [_rec(i + 100, "langgraph", "tool_use", "baseline") for i in range(30)]
        state = self._make(
            tmp_path,
            {
                ("langgraph", "tool_use", "baseline"): wrong_ids,
            },
        )
        errors = state.validate_checkpoint(tmp_path, n_runs=30)
        assert any("mismatch" in e.lower() or "missing" in e.lower() for e in errors)

    # --- Identity validation ---

    def test_wrong_identity_detected(self, tmp_path):
        """All 30 records labeled autogen but file belongs to langgraph condition."""
        wrong = [_rec(i, "autogen", "rag", "timeout") for i in range(30)]
        state = self._make(
            tmp_path,
            {
                ("langgraph", "tool_use", "none"): wrong,
            },
        )
        errors = state.validate_checkpoint(tmp_path, n_runs=30)
        assert any("autogen" in e.lower() for e in errors)

    def test_legacy_checkpoint_rejected(self, tmp_path):
        """Old-format checkpoint without triples is rejected."""
        key = "langgraph_tool_use_baseline"
        (tmp_path / "checkpoint.json").write_text(json.dumps({"completed_conditions": [key]}))
        records = [_rec(i, "langgraph", "tool_use", "baseline") for i in range(30)]
        (tmp_path / f"{key}.json").write_text(json.dumps(records))

        state = ExperimentState.load(tmp_path / "checkpoint.json")
        errors = state.validate_checkpoint(tmp_path, n_runs=30)
        assert len(errors) == 1
        assert "legacy" in errors[0].lower()

    # --- Round-trip ---

    def test_triple_roundtrip(self, tmp_path):
        state = ExperimentState()
        state.mark_done("langgraph", "tool_use", "malformed_output")
        state.save(tmp_path / "checkpoint.json")
        loaded = ExperimentState.load(tmp_path / "checkpoint.json")
        key = "langgraph_tool_use_malformed_output"
        assert loaded._condition_triples[key] == ("langgraph", "tool_use", "malformed_output")
