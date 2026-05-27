"""Integration tests for model training."""
import pytest
import numpy as np
from src.model import build_all_models
from src.evaluator import evaluate


class TestModelTraining:
    def test_lr_trains(self, prepared_data):
        from src.model import build_model
        m = build_model("linear_regression")
        m.fit(prepared_data["X_train"], prepared_data["y_train"])
        pred = m.predict(prepared_data["X_test"])
        assert len(pred) == len(prepared_data["y_test"])

    def test_rf_trains(self, prepared_data):
        from src.model import build_model
        m = build_model("random_forest")
        m.fit(prepared_data["X_train"], prepared_data["y_train"])
        pred = m.predict(prepared_data["X_test"])
        assert len(pred) == len(prepared_data["y_test"])

    def test_all_models_predict(self, trained_models, prepared_data):
        for key, pipeline in trained_models.items():
            pred = pipeline.predict(prepared_data["X_test"])
            assert len(pred) == len(prepared_data["y_test"])

    def test_rf_beats_random(self, trained_models, prepared_data):
        rf = trained_models["random_forest"]
        y_pred = rf.predict(prepared_data["X_test"])
        metrics = evaluate(prepared_data["y_test"].values, y_pred)
        assert metrics["r2"] > 0.5

    def test_lr_r2_positive(self, trained_models, prepared_data):
        lr = trained_models["linear_regression"]
        y_pred = lr.predict(prepared_data["X_test"])
        metrics = evaluate(prepared_data["y_test"].values, y_pred)
        assert metrics["r2"] > 0

    def test_predictions_reasonable(self, trained_models, prepared_data):
        for key, pipeline in trained_models.items():
            y_pred = pipeline.predict(prepared_data["X_test"])
            assert all(y_pred > 0), f"{key} has negative predictions"

    def test_evaluate_all_models(self, trained_models, prepared_data):
        for key, pipeline in trained_models.items():
            y_pred = pipeline.predict(prepared_data["X_test"])
            metrics = evaluate(prepared_data["y_test"].values, y_pred)
            assert "mae" in metrics
            assert "rmse" in metrics
            assert "r2" in metrics
