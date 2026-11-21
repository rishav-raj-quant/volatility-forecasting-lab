"""Unit tests for RealTimeVolatilityEngine and LiveStreamingSimulator."""

import pytest
import numpy as np
import pandas as pd
from volatility_lab.data.realtime import (
    RealTimeVolatilityEngine,
    LiveStreamingSimulator,
    compute_realtime_forecast_cones,
)


def test_realtime_simulator_step():
    sim = LiveStreamingSimulator(ticker="^NSEI", initial_price=25000.0, baseline_vol_ann=0.15)
    df_init = sim.get_current_dataframe()
    assert len(df_init) == 60
    assert "rolling_rv_ann" in df_init.columns
    assert "ewma_vol_ann" in df_init.columns
    assert "parkinson_ann" in df_init.columns
    assert "garman_klass_ann" in df_init.columns
    assert "rogers_satchell_ann" in df_init.columns
    assert "bipower_var_ann" in df_init.columns
    assert "jump_component" in df_init.columns
    assert "var_95_pct" in df_init.columns
    assert "volatility_regime" in df_init.columns

    # Step simulation
    df_next = sim.step(shock=0.0)
    assert len(df_next) == 61

    # Inject shock
    df_shocked = sim.step(shock=-0.03)
    assert len(df_shocked) == 62
    assert df_shocked["log_return"].iloc[-1] < -0.02


def test_realtime_forecast_cones():
    cones = compute_realtime_forecast_cones(
        current_vol_ann=25.0,
        horizon_bars=20,
        long_run_vol_ann=15.0,
        half_life_bars=10,
    )
    assert len(cones["bars"]) == 20
    assert len(cones["mean_forecast"]) == 20
    assert cones["mean_forecast"][0] > cones["mean_forecast"][-1]  # Mean reverts downward towards 15%
    assert cones["upper_95"][-1] > cones["upper_68"][-1]
    assert cones["lower_95"][-1] < cones["lower_68"][-1]


def test_process_intraday_volatility():
    engine = RealTimeVolatilityEngine(annualize=True)
    dates = pd.date_range("2026-11-20 09:15", periods=50, freq="1min")
    np.random.seed(42)
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.001, 50)))
    df_sample = pd.DataFrame({
        "Open": prices * 0.999,
        "High": prices * 1.002,
        "Low": prices * 0.998,
        "Close": prices,
        "Volume": 10000,
    }, index=dates)

    enriched = engine.process_intraday_volatility(df_sample, ticker="TEST", interval="1m")
    assert "ewma_vol_ann" in enriched.columns
    assert "parkinson_ann" in enriched.columns
    assert "garman_klass_ann" in enriched.columns
    assert not enriched["ewma_vol_ann"].isna().any()
