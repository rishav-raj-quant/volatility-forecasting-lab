"""Unit tests for data acquisition, validation, preprocessing, and storage."""

import os
import pytest
import numpy as np
import pandas as pd
from volatility_lab.data.providers.synthetic import SyntheticMarketDataProvider
from volatility_lab.data.validation import DataValidator
from volatility_lab.data.preprocessing import DataPreprocessor
from volatility_lab.data.storage import ParquetStorageManager


@pytest.fixture
def sample_market_data():
    provider = SyntheticMarketDataProvider(seed=42)
    return provider.fetch_data("SYNTHETIC", start_date="2023-01-01", end_date="2023-06-30")


def test_synthetic_data_generation(sample_market_data):
    df = sample_market_data
    assert not df.empty
    assert len(df) > 100
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in df.columns
    assert (df["High"] >= df["Low"]).all()
    assert (df["Close"] > 0).all()


def test_data_validator_valid(sample_market_data):
    validator = DataValidator()
    report = validator.validate(sample_market_data)
    assert report.is_valid
    assert report.invalid_high_low == 0
    assert report.zero_or_negative_prices == 0


def test_data_validator_detects_violations():
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    corrupt_df = pd.DataFrame(
        {
            "Open": [100.0] * 10,
            "High": [90.0] * 10,  # High < Low violation!
            "Low": [110.0] * 10,
            "Close": [-5.0] * 10,  # Negative price!
            "Volume": [1000] * 10,
        },
        index=dates,
    )
    validator = DataValidator()
    report = validator.validate(corrupt_df)
    assert not report.is_valid
    assert report.invalid_high_low > 0
    assert report.zero_or_negative_prices > 0


def test_preprocessor(sample_market_data):
    preprocessor = DataPreprocessor()
    clean = preprocessor.clean_and_prepare(sample_market_data)
    assert "log_return" in clean.columns
    assert "daily_variance_proxy" in clean.columns
    assert not clean["log_return"].isna().any()
    assert (clean["daily_variance_proxy"] >= 0.0).all()


def test_parquet_storage(tmp_path, sample_market_data):
    storage = ParquetStorageManager(base_dir=str(tmp_path))
    preprocessor = DataPreprocessor()
    clean = preprocessor.clean_and_prepare(sample_market_data)

    saved_path = storage.save_processed(clean, "TEST_SYM")
    assert os.path.exists(saved_path)

    loaded = storage.load_processed("TEST_SYM")
    assert loaded is not None
    assert len(loaded) == len(clean)
    assert "log_return" in loaded.columns
