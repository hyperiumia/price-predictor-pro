"""Tests for config module."""
import pytest


def test_app_name():
    from src.config import APP_NAME
    assert APP_NAME == "Price Predictor Pro"


def test_app_version():
    from src.config import APP_VERSION
    assert APP_VERSION == "1.0.0"


def test_app_author():
    from src.config import APP_AUTHOR
    assert "Tirado" in APP_AUTHOR


def test_features_list():
    from src.config import FEATURES
    assert isinstance(FEATURES, list)
    assert len(FEATURES) >= 5


def test_target():
    from src.config import TARGET
    assert TARGET == "price"


def test_default_test_size():
    from src.config import DEFAULT_TEST_SIZE
    assert 0 < DEFAULT_TEST_SIZE < 1


def test_neighborhoods():
    from src.config import NEIGHBORHOODS
    assert isinstance(NEIGHBORHOODS, list)
    assert len(NEIGHBORHOODS) >= 3
