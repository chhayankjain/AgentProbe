"""Tests for experiment metadata capture."""

from __future__ import annotations

import json
import platform

from agentprobe.benchmark.metadata import (
    ExperimentMetadata,
    capture_metadata,
    _get_cpu_model,
    _get_git_info,
    _get_ram_gb,
)


class TestCaptureMetadata:
    """Test the capture_metadata() convenience function."""

    def test_returns_experiment_metadata(self):
        meta = capture_metadata(model="llama3.1")
        assert isinstance(meta, ExperimentMetadata)

    def test_model_name_recorded(self):
        meta = capture_metadata(model="llama3.1", llm_backend="ollama")
        assert meta.model_name == "llama3.1"
        assert meta.llm_backend == "ollama"

    def test_frameworks_default(self):
        meta = capture_metadata(model="test")
        assert set(meta.frameworks) == {"langgraph", "langchain", "autogen"}

    def test_frameworks_custom(self):
        meta = capture_metadata(model="test", frameworks=["langgraph"])
        assert meta.frameworks == ("langgraph",)

    def test_python_version_present(self):
        meta = capture_metadata(model="test")
        assert meta.python_version.startswith("3.")

    def test_cpu_count_positive(self):
        meta = capture_metadata(model="test")
        assert meta.cpu_count > 0

    def test_os_name_matches_platform(self):
        meta = capture_metadata(model="test")
        assert meta.os_name == platform.system()

    def test_seed_recorded(self):
        meta = capture_metadata(model="test", seed=123)
        assert meta.seed == 123

    def test_n_runs_per_cell_recorded(self):
        meta = capture_metadata(model="test", n_runs_per_cell=50)
        assert meta.n_runs_per_cell == 50

    def test_timestamp_utc_present(self):
        meta = capture_metadata(model="test")
        assert meta.timestamp_utc.endswith("Z")
        assert "T" in meta.timestamp_utc

    def test_dependency_versions_has_keys(self):
        meta = capture_metadata(model="test")
        assert "pydantic" in meta.dependency_versions
        assert "numpy" in meta.dependency_versions

    def test_agentprobe_version_present(self):
        meta = capture_metadata(model="test")
        assert meta.agentprobe_version is not None


class TestToDict:
    """Test serialization."""

    def test_to_dict_is_json_serializable(self):
        meta = capture_metadata(model="test")
        d = meta.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_frameworks_is_list_in_dict(self):
        meta = capture_metadata(model="test", frameworks=["langgraph"])
        d = meta.to_dict()
        assert isinstance(d["frameworks"], list)

    def test_all_fields_present_in_dict(self):
        meta = capture_metadata(model="test")
        d = meta.to_dict()
        expected_keys = {
            "cpu_model", "cpu_count", "ram_gb", "os_name", "os_version", "arch",
            "python_version", "agentprobe_version", "git_hash", "git_branch",
            "git_dirty", "llm_backend", "model_name", "ollama_version",
            "frameworks", "n_runs_per_cell", "seed", "timestamp_utc",
            "dependency_versions",
        }
        assert expected_keys.issubset(d.keys())


class TestHelpers:
    """Test internal helper functions."""

    def test_get_git_info_returns_tuple(self):
        commit, branch, dirty = _get_git_info()
        assert isinstance(commit, str)
        assert isinstance(branch, str)
        assert isinstance(dirty, bool)

    def test_get_ram_gb_positive(self):
        ram = _get_ram_gb()
        assert ram >= 0.0

    def test_get_cpu_model_nonempty(self):
        cpu = _get_cpu_model()
        assert len(cpu) > 0


class TestImmutability:
    """Ensure metadata is immutable after capture."""

    def test_frozen_dataclass(self):
        meta = capture_metadata(model="test")
        try:
            meta.model_name = "changed"  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass
