"""Unit tests for tool contract validation."""

import pytest
from pydantic import ValidationError

from core.data import clear_cache as clear_data_cache
from core.modeling import clear_cache as clear_model_cache
from tools.specs import TOOL_REGISTRY
from core.schemas import (
    DataLoadInput,
    EvalInput,
    FitInput,
    ReportInput,
    PipelineConfig,
    EvalOutput,
)
from tools.functions import (
    evaluate_regression_tool,
    fit_linear_regression_tool,
    load_iris_data_tool,
    write_report_tool,
)


class TestToolContractValidation:
    """Tests for tool input/output contract validation."""

    def teardown_method(self):
        """Clear caches after each test."""
        clear_data_cache()
        clear_model_cache()

    def test_load_iris_data_input_validation(self):
        """Test that DataLoadInput is properly validated."""
        # Valid input
        valid_input = DataLoadInput(feature_subset=["petal length (cm)"])
        assert valid_input is not None

        # Invalid: negative test_size
        with pytest.raises(ValidationError):
            DataLoadInput(test_size=-0.1)

        # Invalid: test_size >= 1.0
        with pytest.raises(ValidationError):
            DataLoadInput(test_size=1.0)

    def test_fit_input_validation(self):
        """Test that FitInput is properly validated."""
        # Valid input
        valid_input = FitInput(dataset_id="some_id")
        assert valid_input is not None

    def test_eval_input_validation(self):
        """Test that EvalInput is properly validated."""
        # Valid input
        valid_input = EvalInput(model_id="model_1", dataset_id="dataset_1")
        assert valid_input is not None

    def test_report_input_validation(self):
        """Test that ReportInput is properly validated."""
        # Valid input
        eval_output = EvalOutput(r2=0.9, mae=0.1, rmse=0.15, n_samples=30, verdict="accept")
        valid_input = ReportInput(model_id="model_1", eval=eval_output, attempt=1)
        assert valid_input is not None

        # Invalid: attempt = 0
        with pytest.raises(ValidationError):
            ReportInput(model_id="model_1", eval=eval_output, attempt=0)

    def test_tool_registry_completeness(self):
        """Test that all expected tools are in the registry."""
        expected_tools = {"load_iris_data", "fit_linear_regression", "evaluate_regression", "write_report"}
        assert set(TOOL_REGISTRY.keys()) == expected_tools

    def test_tool_spec_has_valid_function(self):
        """Test that each ToolSpec's fn is callable."""
        for name, spec in TOOL_REGISTRY.items():
            assert callable(spec.fn), f"Tool {name} has non-callable fn"

    def test_load_iris_data_output_round_trip(self):
        """Test that load_iris_data input/output schemas round-trip."""
        input_config = DataLoadInput(feature_subset=["petal length (cm)"])
        output = load_iris_data_tool(input_config)

        # Verify output can be serialized and deserialized
        output_dict = output.model_dump()
        output_reconstructed = TOOL_REGISTRY["load_iris_data"].output_model(**output_dict)
        assert output_reconstructed.n_train == output.n_train
        assert output_reconstructed.dataset_id == output.dataset_id

    def test_malformed_input_produces_validation_error(self):
        """Test that tools gracefully reject malformed input."""
        # Missing required field in DataLoadInput (feature_subset has a default, but let's test another way)
        # Actually, let's test with a completely invalid type
        with pytest.raises((ValidationError, TypeError)):
            DataLoadInput.model_validate({"feature_subset": "not_a_list"})

    def test_tool_spec_input_output_models_are_pydantic(self):
        """Test that all ToolSpec input/output models are Pydantic."""
        for name, spec in TOOL_REGISTRY.items():
            assert hasattr(spec.input_model, "model_validate"), f"{name} input_model is not Pydantic"
            assert hasattr(spec.output_model, "model_validate"), f"{name} output_model is not Pydantic"
