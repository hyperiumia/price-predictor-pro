"""Model evaluation metrics."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true, y_pred) -> dict:
    """Calculate all metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "mape": round(mape, 2),
    }


def compare_models(results: dict) -> str:
    """Return best model key based on RMSE."""
    best_key = min(results, key=lambda k: results[k]["metrics"]["rmse"])
    return best_key
