# Price Predictor Pro

> Machine Learning price prediction with Linear Regression and Random Forest. Built in Python.

## Features

- **Two ML Models** -- Linear Regression and Random Forest Regressor
- **Feature Engineering** -- StandardScaler, OneHotEncoding, ColumnTransformer pipeline
- **Full Metrics** -- MAE, RMSE, R2 Score, MAPE
- **Visualizations** -- Actual vs Predicted, Residuals, Feature Importance, Model Comparison
- **Synthetic Data Generator** -- 500 sample housing records with realistic correlations
- **57 Unit Tests** -- all passing

## Quick Start

```
git clone https://github.com/hyperiumia/price-predictor-pro.git
cd price-predictor-pro
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```
# Run demo with sample data
python main.py demo

# Evaluate models on CSV
python main.py evaluate data.csv

# Show info
python main.py info
```

## Models

| Model | Algorithm | Key Params |
|-------|-----------|------------|
| Linear Regression | Ordinary Least Squares | -- |
| Random Forest | Ensemble of Decision Trees | n_estimators=100 |

## Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R2 Score | Coefficient of Determination |
| MAPE | Mean Absolute Percentage Error |

## Project Structure

```
src/
  config.py         # App configuration
  data_loader.py    # Data generation and CSV loading
  feature_engine.py # Feature preprocessing pipeline
  model.py          # Model builders (LR, RF)
  evaluator.py      # Metrics and comparison
  visualizer.py     # matplotlib charts
  display.py        # Rich terminal output
  cli.py            # CLI interface
tests/
  conftest.py       # Shared fixtures
  test_config.py    # 7 tests
  test_data_loader.py    # 14 tests
  test_feature_engine.py # 10 tests
  test_model.py          # 9 tests
  test_evaluator.py      # 10 tests
  test_model_training.py # 7 tests
```

## Tests

```
pytest tests/ -v
# 57 passed
```

## License

MIT