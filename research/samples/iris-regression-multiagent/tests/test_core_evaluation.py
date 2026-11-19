"""Unit tests for core.evaluation (deterministic, no LLM)."""

import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core.data import clear_cache as clear_data_cache, get_dataset, load_iris_data
from core.evaluation import evaluate_regression
from core.modeling import clear_cache as clear_model_cache, fit_linear_regression
from core.schemas import DataLoadInput, EvalInput, FitInput, PipelineConfig, DEFAULT_FEATURES


class TestEvaluateRegression:
    """Tests for evaluate_regression function."""

    def setup_method(self):
        """Load and fit a model before each test."""
        self.config = PipelineConfig()

        # Load dataset
        input_config = DataLoadInput(
            feature_subset=DEFAULT_FEATURES,
            test_size=0.2,
            random_state=42,
        )
        data_output = load_iris_data(input_config)
        self.dataset_id = data_output.dataset_id

        # Fit model
        fit_input = FitInput(dataset_id=self.dataset_id)
        fit_output = fit_linear_regression(fit_input)
        self.model_id = fit_output.model_id

    def teardown_method(self):
        """Clear caches after each test."""
        clear_data_cache()
        clear_model_cache()

    def test_evaluate_returns_valid_output(self):
        """Test that evaluate returns a valid EvalOutput."""
        eval_input = EvalInput(model_id=self.model_id, dataset_id=self.dataset_id)
        output = evaluate_regression(eval_input, self.config)

        assert 0.0 <= output.r2 <= 1.0
        assert output.mae >= 0.0
        assert output.rmse >= 0.0
        assert output.n_samples > 0
        assert output.verdict in ["accept", "retrain"]

    def test_metrics_match_sklearn(self):
        """Test that computed metrics match sklearn.metrics."""
        from core.modeling import get_model

        eval_input = EvalInput(model_id=self.model_id, dataset_id=self.dataset_id)
        output = evaluate_regression(eval_input, self.config)

        # Get model and data for manual verification
        model, _ = get_model(self.model_id)
        X_train, X_test, y_train, y_test, _ = get_dataset(self.dataset_id)

        # Compute metrics using sklearn directly
        y_pred = model.predict(X_test)
        expected_r2 = r2_score(y_test, y_pred)
        expected_mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        expected_rmse = float(mse ** 0.5)

        # Should match within floating-point precision
        assert abs(output.r2 - expected_r2) < 1e-10
        assert abs(output.mae - expected_mae) < 1e-10
        assert abs(output.rmse - expected_rmse) < 1e-10

    def test_verdict_accept_when_r2_meets_threshold(self):
        """Test that verdict is 'accept' when R² >= threshold."""
        # 3-feature model should have high R² (>= 0.85)
        eval_input = EvalInput(model_id=self.model_id, dataset_id=self.dataset_id)
        output = evaluate_regression(eval_input, self.config)

        # The 3-feature model is known to have R² > 0.9
        assert output.r2 >= self.config.r2_threshold
        assert output.verdict == "accept"

    def test_verdict_retrain_when_r2_below_threshold(self):
        """Test that verdict is 'retrain' when R² < threshold."""
        # Fit a single-feature model with a very high threshold so it's rejected
        clear_data_cache()
        clear_model_cache()

        single_feature_input = DataLoadInput(
            feature_subset=["sepal width (cm)"],  # This feature has lower correlation with target
            test_size=0.2,
            random_state=42,
        )
        data_output = load_iris_data(single_feature_input)

        fit_input = FitInput(dataset_id=data_output.dataset_id)
        fit_output = fit_linear_regression(fit_input)

        # Evaluate with very high threshold (all single-feature models will fail this)
        config = PipelineConfig(r2_threshold=0.99)
        eval_input = EvalInput(model_id=fit_output.model_id, dataset_id=data_output.dataset_id)
        output = evaluate_regression(eval_input, config)

        # Should have low R² (< 0.99) and get retrain verdict
        assert output.r2 < config.r2_threshold
        assert output.verdict == "retrain"

    def test_verdict_changes_at_threshold_boundary(self):
        """Test that verdict flips at the threshold value."""
        eval_input = EvalInput(model_id=self.model_id, dataset_id=self.dataset_id)
        output = evaluate_regression(eval_input, self.config)

        # Create a config with threshold just above actual R²
        config_reject = PipelineConfig(r2_threshold=output.r2 + 0.01)
        eval_reject = evaluate_regression(eval_input, config_reject)
        assert eval_reject.verdict == "retrain"

        # Create a config with threshold just below actual R²
        config_accept = PipelineConfig(r2_threshold=output.r2 - 0.01)
        eval_accept = evaluate_regression(eval_input, config_accept)
        assert eval_accept.verdict == "accept"

    def test_invalid_model_id_raises_key_error(self):
        """Test that invalid model_id raises KeyError."""
        eval_input = EvalInput(model_id="nonexistent_model", dataset_id=self.dataset_id)

        with pytest.raises(KeyError):
            evaluate_regression(eval_input, self.config)

    def test_invalid_dataset_id_raises_key_error(self):
        """Test that invalid dataset_id raises KeyError."""
        eval_input = EvalInput(model_id=self.model_id, dataset_id="nonexistent_dataset")

        with pytest.raises(KeyError):
            evaluate_regression(eval_input, self.config)
