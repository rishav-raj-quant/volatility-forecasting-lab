"""Unit tests for machine learning volatility models."""

import pytest
import numpy as np
import pandas as pd
from volatility_lab.data.providers.synthetic import SyntheticMarketDataProvider
from volatility_lab.data.preprocessing import DataPreprocessor
from volatility_lab.features.feature_pipeline import FeaturePipeline
from volatility_lab.machine_learning.registry import MLModelRegistry
from volatility_lab.machine_learning.quantile import VolatilityQuantileForecaster


@pytest.fixture
def dataset():
    raw = SyntheticMarketDataProvider(seed=42).fetch_data("SYNTHETIC", "2022-01-01", "2023-12-31")
    clean = DataPreprocessor().clean_and_prepare(raw)
    pipe = FeaturePipeline()
    X, y = pipe.transform(clean, horizon=1)
    return X, y


@pytest.mark.parametrize("model_name", [
    "ridge",
    "lasso",
    "elastic_net",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "lightgbm",
    "svr",
])
def test_ml_models_fit_predict(dataset, model_name):
    X, y = dataset
    model = MLModelRegistry.get(model_name)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(X)
    assert not np.isnan(preds).any()


def test_quantile_forecaster(dataset):
    X, y = dataset
    qf = VolatilityQuantileForecaster(quantiles=[0.10, 0.50, 0.90], n_estimators=20)
    qf.fit(X, y)
    preds_df = qf.predict(X)

    assert "q_10" in preds_df.columns
    assert "q_50" in preds_df.columns
    assert "q_90" in preds_df.columns

    # Median forecast should generally lie between 10th and 90th percentiles
    assert (preds_df["q_10"] <= preds_df["q_90"]).mean() > 0.90
