"""Shared fixtures."""
import pytest
import pandas as pd
import numpy as np
from src.data_loader import generate_sample_data
from src.feature_engine import prepare_data
from src.model import build_all_models


@pytest.fixture
def sample_df():
    return generate_sample_data(n_samples=100, random_state=42)


@pytest.fixture
def prepared_data(sample_df):
    return prepare_data(sample_df, random_state=42)


@pytest.fixture
def trained_models(prepared_data):
    models = build_all_models()
    for key, pipeline in models.items():
        pipeline.fit(prepared_data["X_train"], prepared_data["y_train"])
    return models
