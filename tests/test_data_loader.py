"""Tests for data_loader module."""
import pytest
import pandas as pd
from pathlib import Path
from src.data_loader import generate_sample_data, load_csv, load_or_generate


class TestGenerateSampleData:
    def test_returns_dataframe(self):
        df = generate_sample_data()
        assert isinstance(df, pd.DataFrame)

    def test_default_rows(self):
        df = generate_sample_data()
        assert len(df) == 500

    def test_custom_rows(self):
        df = generate_sample_data(n_samples=100)
        assert len(df) == 100

    def test_has_all_columns(self):
        df = generate_sample_data()
        expected = {"square_feet", "bedrooms", "bathrooms", "age",
                    "distance_center", "has_garage", "neighborhood", "price"}
        assert expected.issubset(set(df.columns))

    def test_positive_prices(self, sample_df):
        assert (sample_df["price"] > 0).all()

    def test_bedrooms_range(self, sample_df):
        assert sample_df["bedrooms"].between(1, 5).all()

    def test_bathrooms_range(self, sample_df):
        assert sample_df["bathrooms"].between(1, 3).all()

    def test_deterministic(self):
        df1 = generate_sample_data(random_state=42)
        df2 = generate_sample_data(random_state=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds(self):
        df1 = generate_sample_data(random_state=1)
        df2 = generate_sample_data(random_state=2)
        assert not df1["price"].equals(df2["price"])

    def test_neighborhood_valid(self, sample_df):
        from src.config import NEIGHBORHOODS
        assert sample_df["neighborhood"].isin(NEIGHBORHOODS).all()


class TestLoadCSV:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_csv("nonexistent.csv")

    def test_load_csv(self, tmp_path, sample_df):
        path = tmp_path / "test.csv"
        sample_df.to_csv(path, index=False)
        loaded = load_csv(str(path))
        assert len(loaded) == len(sample_df)


class TestLoadOrGenerate:
    def test_generate_when_no_path(self):
        df = load_or_generate()
        assert len(df) == 500

    def test_load_when_path(self, tmp_path, sample_df):
        path = tmp_path / "test.csv"
        sample_df.to_csv(path, index=False)
        loaded = load_or_generate(str(path))
        assert len(loaded) == len(sample_df)
