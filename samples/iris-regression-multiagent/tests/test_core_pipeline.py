"""Unit tests for core.pipeline (end-to-end deterministic oracle run)."""

import pytest

from core.data import clear_cache as clear_data_cache
from core.modeling import clear_cache as clear_model_cache
from core.pipeline import run_pipeline
from core.schemas import PipelineConfig, PipelineRunResult


class TestRunPipeline:
    """Tests for run_pipeline (oracle/ground-truth)."""

    def teardown_method(self):
        """Clear caches after each test."""
        clear_data_cache()
        clear_model_cache()

    def test_pipeline_runs_end_to_end(self):
        """Test that pipeline completes without exception."""
        result = run_pipeline()

        assert isinstance(result, PipelineRunResult)
        assert result.success is True
        assert result.final_r2 is not None
        assert result.attempts > 0
        assert result.report_markdown is not None

    def test_pipeline_achieves_acceptance_threshold(self):
        """Test that pipeline's final model meets the R² threshold."""
        result = run_pipeline()

        assert result.success is True
        assert result.final_r2 >= 0.85  # Default threshold

    def test_pipeline_takes_expected_attempts(self):
        """Test that pipeline requires expected number of retrain attempts.

        With default feature escalation [["petal length"], all 3 features]:
        - Attempt 1: single feature -> R² < 0.85 -> retrain
        - Attempt 2: all 3 features -> R² >= 0.85 -> accept
        So we expect attempts == 2.
        """
        result = run_pipeline()

        assert result.attempts == 2

    def test_pipeline_is_deterministic(self):
        """Test that multiple runs produce identical results."""
        result1 = run_pipeline()
        clear_data_cache()
        clear_model_cache()

        result2 = run_pipeline()

        assert result1.success == result2.success
        assert result1.final_r2 == result2.final_r2
        assert result1.attempts == result2.attempts
        # Report should be identical (same coefficients, same R², etc.)
        assert result1.report_markdown == result2.report_markdown

    def test_pipeline_with_custom_config(self):
        """Test that pipeline respects custom configuration."""
        config = PipelineConfig(
            r2_threshold=0.95,  # Very high threshold
            max_retrain_attempts=1,
            feature_escalation=[["petal length (cm)"], ["sepal length (cm)", "sepal width (cm)", "petal length (cm)"]],
        )
        result = run_pipeline(config)

        # With threshold=0.95 and limited features, might not reach acceptance
        # but should still run without error
        assert isinstance(result, PipelineRunResult)
        assert result.attempts <= config.max_retrain_attempts

    def test_pipeline_output_has_valid_markdown(self):
        """Test that report_markdown is valid content."""
        result = run_pipeline()

        assert result.report_markdown is not None
        assert "# Iris Linear Regression Report" in result.report_markdown
        assert "petal width (cm)" in result.report_markdown
        assert "R²" in result.report_markdown or "R2" in result.report_markdown

    def test_pipeline_respects_max_retrain_attempts(self):
        """Test that pipeline stops at max_retrain_attempts."""
        config = PipelineConfig(
            r2_threshold=0.99,  # Impossible threshold
            max_retrain_attempts=1,
            feature_escalation=[
                ["petal length (cm)"],
                ["sepal length (cm)", "sepal width (cm)", "petal length (cm)"],
                ["sepal length (cm)", "sepal width (cm)", "petal length (cm)"],  # Duplicate, shouldn't reach
            ],
        )
        result = run_pipeline(config)

        assert result.attempts == 1  # Only 1 attempt allowed

    def test_pipeline_handles_empty_feature_escalation(self):
        """Test that pipeline handles empty feature escalation gracefully."""
        config = PipelineConfig(feature_escalation=[])
        result = run_pipeline(config)

        # Should fail gracefully
        assert result.success is False
        assert result.error is not None
        assert result.attempts == 0
