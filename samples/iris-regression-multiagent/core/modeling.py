"""Linear regression modeling for Iris dataset.

Note: Module-level cache (_MODEL_CACHE) is not thread-safe.
See core/data.py for details.
"""

from __future__ import annotations

import hashlib

import numpy as np
from sklearn.linear_model import LinearRegression

from .data import get_dataset
from .schemas import FitInput, FitOutput


# In-process cache: model_id -> (model: LinearRegression, feature_names: list[str])
_MODEL_CACHE: dict[str, tuple[LinearRegression, list[str]]] = {}


def _hash_model_config(dataset_id: str, feature_names: list[str]) -> str:
    """Deterministic hash of a model configuration for cache key."""
    key = f"{dataset_id}|{'|'.join(feature_names)}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def fit_linear_regression(input_config: FitInput) -> FitOutput:
    """Fit a linear regression model on the dataset.

    Args:
        input_config: FitInput with dataset_id.

    Returns:
        FitOutput with coefficients, intercept, and model_id cache key.

    Raises:
        KeyError: If dataset_id not found.
    """
    X_train, X_test, y_train, y_test, feature_names = get_dataset(input_config.dataset_id)

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Cache
    model_id = _hash_model_config(input_config.dataset_id, list(feature_names))
    _MODEL_CACHE[model_id] = (model, list(feature_names))

    return FitOutput(
        coefficients=model.coef_.tolist(),
        intercept=float(model.intercept_),
        feature_names=list(feature_names),
        model_id=model_id,
    )


def get_model(model_id: str) -> tuple[LinearRegression, list[str]]:
    """Retrieve fitted model from cache.

    Raises:
        KeyError: If model_id not found.
    """
    return _MODEL_CACHE[model_id]


def clear_cache() -> None:
    """Clear in-process model cache (for testing/reset)."""
    _MODEL_CACHE.clear()
