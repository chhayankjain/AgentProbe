"""Unit tests for core.data (deterministic, no LLM)."""

import pytest

from core.data import clear_cache, get_dataset, load_iris_data
from core.schemas import DataLoadInput, DEFAULT_FEATURES


class TestLoadIrisData:
    """Tests for load_iris_data function."""

    def teardown_method(self):
        """Clear cache after each test."""
        clear_cache()

    def test_load_with_default_features(self):
        """Test loading with default feature set."""
        input_config = DataLoadInput(
            feature_subset=DEFAULT_FEATURES,
            test_size=0.2,
            random_state=42,
        )
        output = load_iris_data(input_config)

        assert output.n_train > 0
        assert output.n_test > 0
        assert output.n_train + output.n_test == 150  # Iris has 150 samples
        assert output.feature_names == DEFAULT_FEATURES
        assert output.target_name == "petal width (cm)"
        assert len(output.dataset_id) > 0

    def test_load_with_single_feature(self):
        """Test loading with single feature."""
        input_config = DataLoadInput(
            feature_subset=["petal length (cm)"],
            test_size=0.2,
            random_state=42,
        )
        output = load_iris_data(input_config)

        assert output.n_train > 0
        assert output.n_test > 0
        assert output.feature_names == ["petal length (cm)"]

    def test_seeded_split_is_reproducible(self):
        """Test that seeded split produces identical splits."""
        input_config = DataLoadInput(
            feature_subset=DEFAULT_FEATURES,
            test_size=0.2,
            random_state=42,
        )
        output1 = load_iris_data(input_config)
        dataset1 = get_dataset(output1.dataset_id)

        # Reload with same config
        input_config2 = DataLoadInput(
            feature_subset=DEFAULT_FEATURES,
            test_size=0.2,
            random_state=42,
        )
        output2 = load_iris_data(input_config2)
        dataset2 = get_dataset(output2.dataset_id)

        # Should produce the same dataset_id due to same config hash
        # (both load and cache with the same hash)
        assert output1.dataset_id == output2.dataset_id

    def test_different_seeds_produce_different_splits(self):
        """Test that different seeds produce different splits."""
        input_config1 = DataLoadInput(
            feature_subset=DEFAULT_FEATURES,
            test_size=0.2,
            random_state=42,
        )
        output1 = load_iris_data(input_config1)

        input_config2 = DataLoadInput(
            feature_subset=DEFAULT_FEATURES,
            test_size=0.2,
            random_state=99,
        )
        clear_cache()  # Clear to force new load
        output2 = load_iris_data(input_config2)

        # Different seeds should produce different cache keys
        assert output1.dataset_id != output2.dataset_id

    def test_invalid_feature_raises_value_error(self):
        """Test that requesting a non-existent feature raises ValueError."""
        input_config = DataLoadInput(
            feature_subset=["nonexistent feature"],
            test_size=0.2,
            random_state=42,
        )

        with pytest.raises(ValueError, match="not in Iris dataset"):
            load_iris_data(input_config)

    def test_cache_retrieval(self):
        """Test that loaded dataset can be retrieved from cache."""
        input_config = DataLoadInput(
            feature_subset=DEFAULT_FEATURES,
            test_size=0.2,
            random_state=42,
        )
        output = load_iris_data(input_config)

        X_train, X_test, y_train, y_test, feature_names = get_dataset(output.dataset_id)

        assert X_train.shape[0] == output.n_train
        assert X_test.shape[0] == output.n_test
        assert X_train.shape[1] == len(DEFAULT_FEATURES)
        assert feature_names == DEFAULT_FEATURES

    def test_clear_cache_removes_datasets(self):
        """Test that clear_cache removes all cached datasets."""
        input_config = DataLoadInput(
            feature_subset=DEFAULT_FEATURES,
            test_size=0.2,
            random_state=42,
        )
        output = load_iris_data(input_config)
        dataset_id = output.dataset_id

        # Should be retrievable before clearing
        get_dataset(dataset_id)

        clear_cache()

        # Should not be retrievable after clearing
        with pytest.raises(KeyError):
            get_dataset(dataset_id)
