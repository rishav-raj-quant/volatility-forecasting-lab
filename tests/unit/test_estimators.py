"""Unit tests for statistical and range-based volatility estimators."""

import pytest
import numpy as np
import pandas as pd
from volatility_lab.estimators.historical import HistoricalVolatilityEstimator
from volatility_lab.estimators.ewma import EWMAVolatilityEstimator
from volatility_lab.estimators.parkinson import ParkinsonEstimator
from volatility_lab.estimators.garman_klass import GarmanKlassEstimator
from volatility_lab.estimators.rogers_satchell import RogersSatchellEstimator
from volatility_lab.estimators.realized import RealizedVolatilityEstimator
from volatility_lab.utils.calendar import SQRT_252


@pytest.fixture
def ohlc_series():
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    np.random.seed(42)
    close = 10000.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    open_ = close * (1.0 + np.random.normal(0, 0.002, 100))
    high = np.maximum(open_, close) * (1.0 + np.abs(np.random.normal(0, 0.005, 100)))
    low = np.minimum(open_, close) * (1.0 - np.abs(np.random.normal(0, 0.005, 100)))
    returns = pd.Series(np.log(close[1:] / close[:-1]), index=dates[1:])
    return pd.DataFrame({"Open": open_[1:], "High": high[1:], "Low": low[1:], "Close": close[1:]}, index=dates[1:]), returns


def test_historical_volatility(ohlc_series):
    _, returns = ohlc_series
    est = HistoricalVolatilityEstimator(windows=20, annualize=True)
    res = est.estimate(returns)

    assert "hist_vol_20d" in res.columns
    # Check that estimated volatility is positive and reasonable (e.g. 5% - 40%)
    valid = res["hist_vol_20d"].dropna()
    assert (valid > 0.0).all()
    assert (valid < 1.0).all()


def test_ewma_volatility(ohlc_series):
    _, returns = ohlc_series
    est = EWMAVolatilityEstimator(decay_factor=0.94, annualize=True)
    res = est.estimate(returns)

    assert len(res) == len(returns)
    assert (res > 0.0).all()


def test_parkinson_estimator(ohlc_series):
    ohlc, _ = ohlc_series
    est = ParkinsonEstimator(window=20, annualize=True)
    res = est.estimate(ohlc["High"], ohlc["Low"])

    valid = res.dropna()
    assert (valid > 0.0).all()


def test_garman_klass_estimator(ohlc_series):
    ohlc, _ = ohlc_series
    est = GarmanKlassEstimator(window=20, annualize=True)
    res = est.estimate(ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"])

    valid = res.dropna()
    assert (valid > 0.0).all()


def test_rogers_satchell_estimator(ohlc_series):
    ohlc, _ = ohlc_series
    est = RogersSatchellEstimator(window=20, annualize=True)
    res = est.estimate(ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"])

    valid = res.dropna()
    assert (valid > 0.0).all()


def test_realized_volatility():
    # Synthetic intraday returns
    intraday_dates = pd.date_range("2023-01-01 09:15", "2023-01-05 15:30", freq="5min")
    np.random.seed(42)
    intraday_returns = pd.Series(np.random.normal(0, 0.001, len(intraday_dates)), index=intraday_dates)

    est = RealizedVolatilityEstimator(annualize=True)
    df_rv = est.from_intraday_returns(intraday_returns)

    assert not df_rv.empty
    assert "realized_variance" in df_rv.columns
    assert "bipower_variation" in df_rv.columns
    assert "jump_component" in df_rv.columns
    assert (df_rv["realized_variance"] >= 0.0).all()
    assert (df_rv["jump_component"] >= 0.0).all()
