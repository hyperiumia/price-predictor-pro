"""Tests for feature_engine module."""
import pytest
import numpy as np
from src.feature_engine import (
    build_preprocessor, get_feature_names, split_data, prepare_data,
)


class TestFeatureNames:
    def test_returns_dict(self):
        names = get_feature_names()
        assert isinstance(names, dict)

    def test_has_numeric(self):
        names = get_feature_names()
        assert "square_feet" in names["numeric"]

    def test_has_categorical(self):
        names = get_feature_names()
        assert "neighborhood" in names["categorical"]


class TestSplitData:
    def test_split_shapes(self, sample_df):
        X_train, X_test, y_train, y_test = split_data(sample_df)
        assert len(X_train) + len(X_test) == len(sample_df)

    def test_split_ratio(self, sample_df):
        X_train, X_test, y_train, y_test = split_data(sample_df, test_size=0.3)
        assert abs(len(X_test) / len(sample_df) - 0.3) < 0.05

    def test_deterministic_split(self, sample_df):
        split1 = split_data(sample_df, random_state=42)
        split2 = split_data(sample_df, random_state=42)
        np.testing.assert_array_equal(split1[0].values, split2[0].values)


class TestPreprocessor:
    def test_builds(self):
        preprocessor = build_preprocessor()
        assert preprocessor is not None

    def test_has_transformers(self):
        preprocessor = build_preprocessor()
        assert len(preprocessor.transformers) == 3


class TestPrepareData:
    def test_returns_dict(self, sample_df):
        data = prepare_data(sample_df)
        assert isinstance(data, dict)
        assert "X_train" in data
        assert "y_train" in data

    def test_feature_names_included(self, sample_df):
        data = prepare_data(sample_df)
        assert "feature_names" in data
