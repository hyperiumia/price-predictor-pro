"""Tests for evaluator module."""
import pytest
import numpy as np
from src.evaluator import evaluate, compare_models


class TestEvaluate:
    def test_returns_dict(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        metrics = evaluate(y_true, y_pred)
        assert isinstance(metrics, dict)

    def test_has_all_metrics(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        metrics = evaluate(y_true, y_pred)
        assert all(k in metrics for k in ["mae", "mse", "rmse", "r2", "mape"])

    def test_perfect_prediction(self):
        y = np.array([100, 200, 300])
        metrics = evaluate(y, y)
        assert metrics["mae"] == 0
        assert metrics["rmse"] == 0
        assert metrics["r2"] == 1.0

    def test_mae_positive(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        metrics = evaluate(y_true, y_pred)
        assert metrics["mae"] > 0

    def test_rmse_greater_than_mae(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([150, 150, 350])
        metrics = evaluate(y_true, y_pred)
        assert metrics["rmse"] >= metrics["mae"]

    def test_r2_range(self):
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        metrics = evaluate(y_true, y_pred)
        assert -1 <= metrics["r2"] <= 1

    def test_mape_reasonable(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        metrics = evaluate(y_true, y_pred)
        assert 0 < metrics["mape"] < 100


class TestCompareModels:
    def test_returns_string(self):
        results = {
            "lr": {"metrics": {"rmse": 100}},
            "rf": {"metrics": {"rmse": 50}},
        }
        best = compare_models(results)
        assert isinstance(best, str)

    def test_picks_lowest_rmse(self):
        results = {
            "lr": {"metrics": {"rmse": 100}},
            "rf": {"metrics": {"rmse": 50}},
        }
        assert compare_models(results) == "rf"

    def test_tie(self):
        results = {
            "lr": {"metrics": {"rmse": 100}},
            "rf": {"metrics": {"rmse": 100}},
        }
        best = compare_models(results)
        assert best in ["lr", "rf"]
