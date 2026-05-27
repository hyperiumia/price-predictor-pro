"""App configuration."""

APP_NAME = "Price Predictor Pro"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Patricio Tirado - Hyperium IA"

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

FEATURES = [
    "square_feet", "bedrooms", "bathrooms",
    "age", "distance_center", "has_garage",
    "neighborhood",
]
TARGET = "price"

NEIGHBORHOODS = ["downtown", "suburban", "rural", "industrial"]
