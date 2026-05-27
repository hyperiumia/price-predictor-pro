"""CLI interface for Price Predictor Pro."""
import argparse
import sys
from pathlib import Path
from src.config import APP_NAME, APP_VERSION
from src.data_loader import load_or_generate
from src.feature_engine import prepare_data
from src.model import build_all_models, get_model_name
from src.evaluator import evaluate, compare_models
from src.visualizer import (
    plot_actual_vs_predicted, plot_residuals, plot_model_comparison,
)
from src.display import (
    print_banner, print_info, console,
    print_metrics_table, print_prediction,
)


def cmd_demo(_args):
    """Run demo with sample data."""
    console.print("[dim]Generating sample housing data...[/dim]")
    df = load_or_generate()
    console.print(f"[green]Generated {len(df)} samples[/green]\n")

    data = prepare_data(df)
    console.print(f"Train: {len(data['X_train'])} | Test: {len(data['X_test'])}\n")

    models = build_all_models()
    results = {}

    for key, pipeline in models.items():
        name = get_model_name(key)
        console.print(f"[cyan]Training {name}...[/cyan]")
        pipeline.fit(data["X_train"], data["y_train"])
        y_pred = pipeline.predict(data["X_test"])
        metrics = evaluate(data["y_test"].values, y_pred)

        results[key] = {
            "pipeline": pipeline,
            "predictions": y_pred,
            "metrics": metrics,
        }

        predictions_list = [
            {"actual": a, "predicted": p}
            for a, p in zip(data["y_test"].values[:10], y_pred[:10])
        ]

        plot_actual_vs_predicted(
            data["y_test"].values, y_pred,
            f"{name} - Actual vs Predicted",
            f"assets/{key}_predictions.png",
        )
        plot_residuals(
            data["y_test"].values, y_pred,
            f"{name} - Residuals",
            f"assets/{key}_residuals.png",
        )

    model_names = {k: get_model_name(k) for k in results}
    print_metrics_table(results, model_names)

    best_key = compare_models(results)
    best_name = get_model_name(best_key)
    console.print(f"[green bold]Best model: {best_name}[/green bold]\n")

    best_preds = [
        {"actual": a, "predicted": p}
        for a, p in zip(data["y_test"].values, results[best_key]["predictions"])
    ]
    print_prediction(best_preds)

    console.print("[dim]Charts saved to assets/[/dim]\n")


def cmd_train(args):
    """Train models on CSV data."""
    df = load_or_generate(args.file)
    console.print(f"[green]Loaded {len(df)} samples from {args.file}[/green]")

    data = prepare_data(df)
    models = build_all_models()

    for key, pipeline in models.items():
        name = get_model_name(key)
        console.print(f"[cyan]Training {name}...[/cyan]")
        pipeline.fit(data["X_train"], data["y_train"])

    console.print("[green bold]All models trained successfully[/green bold]\n")


def cmd_evaluate(args):
    """Evaluate and compare models."""
    df = load_or_generate(args.file)
    data = prepare_data(df)
    models = build_all_models()
    results = {}

    for key, pipeline in models.items():
        pipeline.fit(data["X_train"], data["y_train"])
        y_pred = pipeline.predict(data["X_test"])
        metrics = evaluate(data["y_test"].values, y_pred)
        results[key] = {"metrics": metrics, "predictions": y_pred}

    model_names = {k: get_model_name(k) for k in results}
    print_metrics_table(results, model_names)

    best_key = compare_models(results)
    console.print(f"[green bold]Best model: {get_model_name(best_key)}[/green bold]\n")


def cmd_info(_args):
    print_info()


def run(args):
    print_banner()

    if not args.command:
        console.print("[red bold]Error:[/red bold] Choose: demo, train, evaluate, info")
        return

    handlers = {
        "demo": cmd_demo, "d": cmd_demo,
        "train": cmd_train,
        "evaluate": cmd_evaluate, "e": cmd_evaluate,
        "info": cmd_info,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        console.print(f"[red bold]Error:[/red bold] Unknown command: {args.command}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="price-predictor",
        description=f"{APP_NAME} v{APP_VERSION} - ML Price Prediction",
        epilog="Usage: python main.py demo | train FILE | evaluate FILE | info",
    )
    subparsers = ap.add_subparsers(dest="command", help="Command")

    subparsers.add_parser("demo", aliases=["d"], help="Run demo with sample data")

    train_p = subparsers.add_parser("train", help="Train models on CSV")
    train_p.add_argument("file", help="CSV data file")

    eval_p = subparsers.add_parser("evaluate", aliases=["e"], help="Evaluate models")
    eval_p.add_argument("file", nargs="?", help="CSV data file")

    subparsers.add_parser("info", help="Show app info")

    return ap


if __name__ == "__main__":
    ap = build_parser()
    args = ap.parse_args()
    run(args)
