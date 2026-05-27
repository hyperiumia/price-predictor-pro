"""Visualization with matplotlib."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_actual_vs_predicted(y_true, y_pred, title: str, output_path: str):
    """Scatter plot: actual vs predicted."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="navy", facecolors="cornflowerblue", s=30)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction")

    ax.set_xlabel("Actual Price", fontsize=12)
    ax.set_ylabel("Predicted Price", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(feature_names, importances, title: str, output_path: str):
    """Bar chart of feature importances."""
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
    ax.barh(range(len(indices)), importances[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_residuals(y_true, y_pred, title: str, output_path: str):
    """Residuals distribution."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors="navy", facecolors="cornflowerblue", s=30)
    axes[0].axhline(y=0, color="r", linestyle="--")
    axes[0].set_xlabel("Predicted Price")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=30, edgecolor="black", alpha=0.7, color="cornflowerblue")
    axes[1].axvline(x=0, color="r", linestyle="--")
    axes[1].set_xlabel("Residual Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residuals Distribution")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_comparison(model_names: dict, metrics: dict, metric_name: str, output_path: str):
    """Bar chart comparing models on a metric."""
    names = [model_names[k] for k in metrics]
    values = [metrics[k] for k in metrics]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=["cornflowerblue", "coral"], edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:,.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"Model Comparison - {metric_name}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
