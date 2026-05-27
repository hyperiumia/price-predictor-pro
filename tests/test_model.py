"""Tests for model module."""
import pytest
from src.model import build_model, build_all_models, get_model_name, MODELS


class TestBuildModel:
    def test_linear_regression(self):
        m = build_model("linear_regression")
        assert m is not None

    def test_random_forest(self):
        m = build_model("random_forest")
        assert m is not None

    def test_unknown_model(self):
        with pytest.raises(ValueError):
            build_model("unknown")

    def test_has_preprocessor(self):
        m = build_model("linear_regression")
        assert "preprocessor" in m.named_steps

    def test_has_model_step(self):
        m = build_model("random_forest")
        assert "model" in m.named_steps


class TestBuildAllModels:
    def test_returns_all(self):
        models = build_all_models()
        assert len(models) == len(MODELS)

    def test_keys_match(self):
        models = build_all_models()
        assert set(models.keys()) == set(MODELS.keys())


class TestGetModelName:
    def test_valid_key(self):
        name = get_model_name("linear_regression")
        assert name == "Linear Regression"

    def test_random_forest(self):
        name = get_model_name("random_forest")
        assert name == "Random Forest"
