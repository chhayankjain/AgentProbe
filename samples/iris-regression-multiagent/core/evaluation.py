"""Evaluation metrics for linear regression models."""

from __future__ import annotations

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import get_dataset
from .modeling import get_model
from .schemas import EvalInput, EvalOutput, PipelineConfig


def evaluate_regression(input_config: EvalInput, config: PipelineConfig) -> EvalOutput:
    """Evaluate fitted model on test set.

    Args:
        input_config: EvalInput with model_id and dataset_id.
        config: PipelineConfig for r2_threshold.

    Returns:
        EvalOutput with R², MAE, RMSE, and accept/retrain verdict.

    Raises:
        KeyError: If model_id or dataset_id not found.
    """
    # Retrieve model and data
    model, _ = get_model(input_config.model_id)
    X_train, X_test, y_train, y_test, _ = get_dataset(input_config.dataset_id)

    # Compute metrics on test set
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(mse ** 0.5)

    # Deterministic verdict based on threshold
    verdict = "accept" if r2 >= config.r2_threshold else "retrain"

    return EvalOutput(
        r2=r2,
        mae=mae,
        rmse=rmse,
        n_samples=len(X_test),
        verdict=verdict,
    )
