"""Pydantic schemas for Iris regression pipeline I/O — framework-agnostic contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# Feature and target constants (framework-agnostic, locked for all runs)
DEFAULT_FEATURES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)"]
DEFAULT_TARGET = "petal width (cm)"


class DataLoadInput(BaseModel):
    """Input contract for load_iris_data tool."""
    feature_subset: list[str] = Field(default_factory=lambda: list(DEFAULT_FEATURES))
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0, description="Train/test split ratio")
    random_state: int = Field(default=42, description="Seed for reproducible split")


class DataLoadOutput(BaseModel):
    """Output contract for load_iris_data tool.

    Note: dataset arrays are kept in-process cache (dataset_id references them).
    This avoids sending large arrays into LLM context.
    """
    n_train: int = Field(description="Number of training samples")
    n_test: int = Field(description="Number of test samples")
    feature_names: list[str] = Field(description="Names of input features")
    target_name: str = Field(default=DEFAULT_TARGET, description="Name of target variable")
    dataset_id: str = Field(description="In-process cache key for train/test arrays")


class FitInput(BaseModel):
    """Input contract for fit_linear_regression tool."""
    dataset_id: str = Field(description="Reference to dataset in in-process cache")


class FitOutput(BaseModel):
    """Output contract for fit_linear_regression tool."""
    coefficients: list[float] = Field(description="Linear regression coefficients (one per feature)")
    intercept: float = Field(description="Linear regression intercept")
    feature_names: list[str] = Field(description="Feature names used in the fit")
    model_id: str = Field(description="In-process cache key for fitted model")


class EvalInput(BaseModel):
    """Input contract for evaluate_regression tool."""
    model_id: str = Field(description="Reference to fitted model in in-process cache")
    dataset_id: str = Field(description="Reference to dataset in in-process cache")


class EvalOutput(BaseModel):
    """Output contract for evaluate_regression tool.

    Verdict is deterministically computed based on r2 vs. PipelineConfig.r2_threshold.
    """
    r2: float = Field(le=1.0, description="R² on test set (can be negative for poor models)")
    mae: float = Field(ge=0.0, description="Mean absolute error on test set")
    rmse: float = Field(ge=0.0, description="Root mean squared error on test set")
    n_samples: int = Field(description="Number of test samples evaluated")
    verdict: Literal["accept", "retrain"] = Field(
        description="accept: R² >= threshold; retrain: R² < threshold"
    )


class ReportInput(BaseModel):
    """Input contract for write_report tool."""
    model_id: str = Field(description="Reference to fitted model")
    eval: EvalOutput = Field(description="Evaluation results (embedded, not cached)")
    attempt: int = Field(ge=1, description="Attempt number (1-indexed)")
    r2_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="R² threshold used for verdict")


class ReportOutput(BaseModel):
    """Output contract for write_report tool."""
    markdown: str = Field(description="Rendered markdown report")


class PipelineConfig(BaseModel):
    """Shared configuration for all pipeline implementations."""
    r2_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Reject if R² < this value")
    max_retrain_attempts: int = Field(
        default=2, ge=1, description="Max number of retrain loops"
    )
    feature_escalation: list[list[str]] = Field(
        default_factory=lambda: [
            ["sepal width (cm)"],  # First attempt: low-correlation feature (will be rejected)
            DEFAULT_FEATURES,  # Second attempt: all 3 features (will be accepted)
        ],
        description="Feature subsets to try in order of escalation",
    )


class PipelineRunResult(BaseModel):
    """Canonical output of every pipeline.run() call (framework-independent).

    This is the contract that adapters map to AgentResult.
    """
    success: bool = Field(description="Whether pipeline completed without unrecovered error")
    final_r2: float | None = Field(
        default=None, description="Final R² of accepted model, or None if rejected"
    )
    attempts: int = Field(ge=0, description="Number of training attempts (0 = initialization error, 1+ = ran)")
    report_markdown: str | None = Field(
        default=None, description="Final markdown report, or None on failure"
    )
    error: str | None = Field(default=None, description="Error message if success=False")
