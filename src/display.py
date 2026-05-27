"""Rich display utilities."""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from src.config import APP_NAME, APP_VERSION, APP_AUTHOR

console = Console()


def print_banner():
    banner = Text()
    banner.append(f"  {APP_NAME}", style="bold cyan")
    banner.append(f" v{APP_VERSION}", style="dim")
    banner.append(f"\n  {APP_AUTHOR}", style="dim cyan")
    console.print(Panel(banner, border_style="cyan", padding=(0, 2)))


def print_metrics_table(results: dict, model_names: dict):
    """Print a Rich table with model metrics."""
    table = Table(title="Model Comparison", border_style="cyan")
    table.add_column("Metric", style="bold")

    for key in results:
        table.add_column(model_names[key], justify="right")

    metrics_keys = ["mae", "rmse", "r2", "mape"]
    labels = {"mae": "MAE", "rmse": "RMSE", "r2": "R2 Score", "mape": "MAPE (%)"}

    for mk in metrics_keys:
        row = [labels[mk]]
        for key in results:
            val = results[key]["metrics"][mk]
            if mk == "r2":
                row.append(f"{val:.4f}")
            elif mk == "mape":
                row.append(f"{val:.2f}%")
            else:
                row.append(f"${val:,.2f}")
        table.add_row(*row)

    console.print()
    console.print(table)
    console.print()


def print_prediction(predictions: list, n: int = 10):
    """Print sample predictions."""
    table = Table(title=f"Sample Predictions (first {n})", border_style="cyan")
    table.add_column("#", style="dim")
    table.add_column("Actual", justify="right")
    table.add_column("Predicted", justify="right")
    table.add_column("Error", justify="right")
    table.add_column("Error %", justify="right")

    for i, p in enumerate(predictions[:n]):
        error = p["actual"] - p["predicted"]
        error_pct = abs(error / p["actual"]) * 100 if p["actual"] != 0 else 0
        color = "green" if abs(error_pct) < 10 else "yellow" if abs(error_pct) < 20 else "red"
        table.add_row(
            str(i + 1),
            f"${p['actual']:,.0f}",
            f"${p['predicted']:,.0f}",
            f"[{color}]{error:+,.0f}[/{color}]",
            f"[{color}]{error_pct:.1f}%[/{color}]",
        )

    console.print()
    console.print(table)
    console.print()


def print_info():
    console.print()
    console.print(f"  [bold]{APP_NAME} v{APP_VERSION}[/bold]")
    console.print(f"  {APP_AUTHOR}\n")
    console.print("  Commands:")
    console.print("  - demo              Run demo with sample data")
    console.print("  - train FILE        Train models on CSV data")
    console.print("  - evaluate FILE     Evaluate and compare models")
    console.print("  - info              Show this help\n")
    console.print("  Models: Linear Regression, Random Forest")
    console.print("  Metrics: MAE, RMSE, R2, MAPE")
    console.print()
