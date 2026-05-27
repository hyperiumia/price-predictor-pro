"""ML models for price prediction."""
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from src.feature_engine import build_preprocessor


MODELS = {
    "linear_regression": {
        "name": "Linear Regression",
        "class": LinearRegression,
        "params": {},
    },
    "random_forest": {
        "name": "Random Forest",
        "class": RandomForestRegressor,
        "params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
    },
}


def build_model(model_key: str) -> Pipeline:
    """Build a sklearn Pipeline with preprocessor + model."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Choose from: {list(MODELS.keys())}")

    config = MODELS[model_key]
    preprocessor = build_preprocessor()
    model = config["class"](**config["params"])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def build_all_models() -> dict:
    """Build all available models."""
    return {key: build_model(key) for key in MODELS}


def get_model_name(key: str) -> str:
    return MODELS[key]["name"]
