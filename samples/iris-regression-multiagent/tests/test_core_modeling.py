"""Unit tests for core.modeling (deterministic, no LLM)."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from core.data import clear_cache as clear_data_cache, load_iris_data
from core.modeling import clear_cache as clear_model_cache, fit_linear_regression, get_model
from core.schemas import DataLoadInput, FitInput, DEFAULT_FEATURES


class TestFitLinearRegression:
    """Tests for fit_linear_regression function."""

    def setup_method(self):
        """Load test dataset before each test."""
        self.input_config = DataLoadInput(
            feature_subset=DEFAULT_FEATURES,
            test_size=0.2,
            random_state=42,
        )
        output = load_iris_data(self.input_config)
        self.dataset_id = output.dataset_id

    def teardown_method(self):
        """Clear caches after each test."""
        clear_data_cache()
        clear_model_cache()

    def test_fit_returns_valid_output(self):
        """Test that fit returns a valid FitOutput."""
        fit_input = FitInput(dataset_id=self.dataset_id)
        output = fit_linear_regression(fit_input)

        assert len(output.coefficients) == len(DEFAULT_FEATURES)
        assert isinstance(output.intercept, float)
        assert output.feature_names == DEFAULT_FEATURES
        assert len(output.model_id) > 0

    def test_coefficients_are_numeric(self):
        """Test that all coefficients are valid floats."""
        fit_input = FitInput(dataset_id=self.dataset_id)
        output = fit_linear_regression(fit_input)

        for coef in output.coefficients:
            assert isinstance(coef, float)
            assert not np.isnan(coef)

    def test_model_is_cached(self):
        """Test that fitted model is cached and retrievable."""
        fit_input = FitInput(dataset_id=self.dataset_id)
        output = fit_linear_regression(fit_input)

        model, feature_names = get_model(output.model_id)
        assert isinstance(model, LinearRegression)
        assert feature_names == DEFAULT_FEATURES

    def test_fit_with_single_feature(self):
        """Test fitting with a single feature."""
        clear_data_cache()
        single_feature_input = DataLoadInput(
            feature_subset=["petal length (cm)"],
            test_size=0.2,
            random_state=42,
        )
        data_output = load_iris_data(single_feature_input)

        fit_input = FitInput(dataset_id=data_output.dataset_id)
        output = fit_linear_regression(fit_input)

        assert len(output.coefficients) == 1
        assert output.feature_names == ["petal length (cm)"]

    def test_seeded_fit_is_reproducible(self):
        """Test that fitting with the same data produces identical coefficients."""
        fit_input = FitInput(dataset_id=self.dataset_id)
        output1 = fit_linear_regression(fit_input)
        coeffs1 = output1.coefficients

        # Refit with same dataset
        clear_model_cache()
        output2 = fit_linear_regression(fit_input)
        coeffs2 = output2.coefficients

        # Coefficients should be identical
        np.testing.assert_array_almost_equal(coeffs1, coeffs2)
        np.testing.assert_almost_equal(output1.intercept, output2.intercept)

    def test_invalid_dataset_id_raises_key_error(self):
        """Test that invalid dataset_id raises KeyError."""
        fit_input = FitInput(dataset_id="nonexistent_dataset_id")

        with pytest.raises(KeyError):
            fit_linear_regression(fit_input)

    def test_clear_cache_removes_models(self):
        """Test that clear_cache removes all cached models."""
        fit_input = FitInput(dataset_id=self.dataset_id)
        output = fit_linear_regression(fit_input)
        model_id = output.model_id

        # Should be retrievable before clearing
        get_model(model_id)

        clear_model_cache()

        # Should not be retrievable after clearing
        with pytest.raises(KeyError):
            get_model(model_id)
