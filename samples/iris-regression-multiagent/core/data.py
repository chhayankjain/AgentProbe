"""Data loading and preprocessing for Iris regression.

Deterministic, seeded, in-process cache to avoid serializing large arrays.
"""

import hashlib
from typing import Any

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from .schemas import DEFAULT_FEATURES, DEFAULT_TARGET, DataLoadInput, DataLoadOutput


# In-process cache: dataset_id -> (X_train, X_test, y_train, y_test, feature_names)
_DATASET_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}


def _hash_config(config: DataLoadInput) -> str:
    """Deterministic hash of a DataLoadInput for cache key."""
    key = f"{config.feature_subset}|{config.test_size}|{config.random_state}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def load_iris_data(input_config: DataLoadInput) -> DataLoadOutput:
    """Load Iris dataset, perform train/test split, return cache key.

    Args:
        input_config: DataLoadInput specifying features, split ratio, random seed.

    Returns:
        DataLoadOutput with shapes and cache key (dataset_id).

    Raises:
        ValueError: If any requested feature does not exist in Iris.
    """
    # Validate feature names
    iris = load_iris(as_frame=True)
    available_features = list(iris.data.columns)
    for feat in input_config.feature_subset:
        if feat not in available_features:
            raise ValueError(
                f"Feature '{feat}' not in Iris dataset. Available: {available_features}"
            )

    # Select features and target
    X = iris.data[input_config.feature_subset].values  # type: ignore
    y = iris.data[DEFAULT_TARGET].values  # type: ignore

    # Train/test split with seed
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=input_config.test_size,
        random_state=input_config.random_state,
    )

    # Cache
    dataset_id = _hash_config(input_config)
    _DATASET_CACHE[dataset_id] = (X_train, X_test, y_train, y_test, input_config.feature_subset)

    return DataLoadOutput(
        n_train=len(X_train),
        n_test=len(X_test),
        feature_names=input_config.feature_subset,
        target_name=DEFAULT_TARGET,
        dataset_id=dataset_id,
    )


def get_dataset(dataset_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Retrieve dataset from cache.

    Raises:
        KeyError: If dataset_id not found.
    """
    return _DATASET_CACHE[dataset_id]


def clear_cache() -> None:
    """Clear in-process dataset cache (for testing/reset)."""
    _DATASET_CACHE.clear()
