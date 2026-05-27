"""Feature engineering pipeline."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.config import FEATURES, TARGET, DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE


NUM_FEATURES = ["square_feet", "bedrooms", "bathrooms", "age", "distance_center"]
CAT_FEATURES = ["neighborhood"]
PASS_FEATURES = ["has_garage"]


def get_feature_names() -> dict:
    return {
        "numeric": NUM_FEATURES,
        "categorical": CAT_FEATURES,
        "passthrough": PASS_FEATURES,
    }


def build_preprocessor() -> ColumnTransformer:
    """Build sklearn ColumnTransformer for feature preprocessing."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT_FEATURES),
            ("pass", "passthrough", PASS_FEATURES),
        ]
    )


def split_data(df: pd.DataFrame, test_size: float = DEFAULT_TEST_SIZE,
               random_state: int = DEFAULT_RANDOM_STATE) -> tuple:
    """Split data into train/test sets."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def prepare_data(df: pd.DataFrame, test_size: float = DEFAULT_TEST_SIZE,
                 random_state: int = DEFAULT_RANDOM_STATE) -> dict:
    """Full pipeline: split and return dict with all data."""
    X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": get_feature_names(),
    }
