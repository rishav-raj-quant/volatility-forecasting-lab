"""Integration tests for the complete end-to-end volatility research pipeline."""

import pytest
import os
import numpy as np
import pandas as pd
from volatility_lab.data.providers.synthetic import SyntheticMarketDataProvider
from volatility_lab.data.preprocessing import DataPreprocessor
from volatility_lab.features.feature_pipeline import FeaturePipeline
from volatility_lab.forecasting.pipeline import VolatilityForecastingPipeline
from volatility_lab.experiments.tracking import ExperimentTracker


def test_full_forecasting_pipeline(tmp_path):
    # 1. Acquire synthetic data
    raw = SyntheticMarketDataProvider(seed=42).fetch_data("SYNTHETIC", "2022-01-01", "2023-12-31")
    assert not raw.empty

    # 2. Preprocessing
    clean = DataPreprocessor().clean_and_prepare(raw)
    assert "log_return" in clean.columns

    # 3. Multi-horizon forecasting
    horizons = [1, 5]
    forecaster = VolatilityForecastingPipeline(horizons=horizons)
    fc_df = forecaster.execute(clean)

    assert not fc_df.empty
    assert "forecast_vol_ann_pct" in fc_df.columns
    assert set(fc_df["horizon_days"].unique()) == set(horizons)

    # 4. Experiment tracking
    tracker = ExperimentTracker(log_dir=str(tmp_path))
    exp_id = tracker.log_run(
        ticker="SYNTHETIC",
        model_name="GARCH(1,1)",
        horizon=1,
        metrics={"qlike": 0.125, "rmse": 0.003},
    )
    assert os.path.exists(tracker.index_file)
    exp_df = tracker.list_experiments()
    assert len(exp_df) == 1
    assert exp_df.iloc[0]["experiment_id"] == exp_id
