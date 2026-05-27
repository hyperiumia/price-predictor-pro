"""Data loading and generation."""
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import FEATURES, TARGET, NEIGHBORHOODS, DEFAULT_RANDOM_STATE


def generate_sample_data(n_samples: int = 500, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Generate synthetic housing data."""
    rng = np.random.RandomState(random_state)

    square_feet = rng.uniform(500, 4000, n_samples).round(0)
    bedrooms = rng.randint(1, 6, n_samples)
    bathrooms = rng.randint(1, 4, n_samples)
    age = rng.randint(0, 50, n_samples)
    distance_center = rng.uniform(0.5, 30, n_samples).round(1)
    has_garage = rng.choice([0, 1], n_samples, p=[0.3, 0.7])
    neighborhood = rng.choice(NEIGHBORHOODS, n_samples)

    base_price = (
        square_feet * 150
        + bedrooms * 10000
        + bathrooms * 8000
        - age * 1500
        - distance_center * 3000
        + has_garage * 20000
    )

    neighborhood_bonus = {
        "downtown": 50000,
        "suburban": 20000,
        "rural": -10000,
        "industrial": -30000,
    }
    bonus = np.array([neighborhood_bonus[n] for n in neighborhood])
    noise = rng.normal(0, 15000, n_samples)
    price = (base_price + bonus + noise).round(0)

    df = pd.DataFrame({
        "square_feet": square_feet.astype(int),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age": age,
        "distance_center": distance_center,
        "has_garage": has_garage,
        "neighborhood": neighborhood,
        "price": price.astype(int),
    })
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(p)


def load_or_generate(path: str = None, n_samples: int = 500) -> pd.DataFrame:
    """Load CSV or generate sample data."""
    if path:
        return load_csv(path)
    return generate_sample_data(n_samples)
