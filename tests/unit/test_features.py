"""Unit tests for feature engineering and strict leakage prevention."""

import pytest
import numpy as np
import pandas as pd
from volatility_lab.data.providers.synthetic import SyntheticMarketDataProvider
from volatility_lab.data.preprocessing import DataPreprocessor
from volatility_lab.features.feature_pipeline import FeaturePipeline
from volatility_lab.features.returns import extract_return_features
from volatility_lab.features.volatility import extract_volatility_features
from volatility_lab.features.technical import extract_technical_features


@pytest.fixture
def clean_data():
    raw = SyntheticMarketDataProvider(seed=42).fetch_data("SYNTHETIC", "2023-01-01", "2023-12-31")
    return DataPreprocessor().clean_and_prepare(raw)


def test_return_features(clean_data):
    ret_feats = extract_return_features(clean_data["log_return"])
    assert "ret_lag_1" in ret_feats.columns
    assert "abs_ret_lag_1" in ret_feats.columns
    assert "sq_ret_lag_1" in ret_feats.columns
    assert "cum_ret_20d" in ret_feats.columns


def test_volatility_features(clean_data):
    vol_feats = extract_volatility_features(clean_data)
    assert "hist_vol_20d" in vol_feats.columns
    assert "ewma_vol_94" in vol_feats.columns
    assert "parkinson_20d" in vol_feats.columns
    assert "garman_klass_20d" in vol_feats.columns


def test_technical_features(clean_data):
    tech_feats = extract_technical_features(clean_data)
    assert "rsi_14" in tech_feats.columns
    assert "atr_norm_14" in tech_feats.columns
    assert "drawdown_20d" in tech_feats.columns


def test_feature_pipeline_leakage_safety(clean_data):
    pipeline = FeaturePipeline()
    X, y = pipeline.transform(clean_data, horizon=5)

    assert not X.empty
    assert not y.empty
    assert len(X) == len(y)

    # Confirm index exact match
    assert (X.index == y.index).all()

    # Leakage check: Target y at index t must match forward 5-day average of clean_data
    test_idx = X.index[50]
    loc_in_clean = clean_data.index.get_loc(test_idx)
    # Future 5 squared returns: loc_in_clean + 1 to loc_in_clean + 5
    expected_y = clean_data["log_return"].iloc[loc_in_clean + 1 : loc_in_clean + 6].pow(2).mean()

    assert np.isclose(y.loc[test_idx], expected_y, atol=1e-7)
