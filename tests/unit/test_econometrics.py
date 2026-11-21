"""Unit tests for econometric volatility models (GARCH, EGARCH, GJR-GARCH, HAR-RV)."""

import pytest
import numpy as np
import pandas as pd
from volatility_lab.data.providers.synthetic import SyntheticMarketDataProvider
from volatility_lab.econometrics.garch import GARCHModel
from volatility_lab.econometrics.egarch import EGARCHModel
from volatility_lab.econometrics.gjr_garch import GJRGARCHModel
from volatility_lab.econometrics.har_rv import HARRVModel


@pytest.fixture
def returns_sample():
    raw = SyntheticMarketDataProvider(seed=42).fetch_data("SYNTHETIC", "2021-01-01", "2023-12-31")
    ret = np.log(raw["Close"] / raw["Close"].shift(1)).dropna()
    return ret, raw["True_Variance"]


def test_garch_model(returns_sample):
    ret, _ = returns_sample
    model = GARCHModel(p=1, q=1)
    res = model.fit(ret)

    assert res.aic is not None
    assert model.persistence > 0.0
    assert model.half_life > 0.0

    # 1-day forecast
    fc_1d = model.forecast(horizon=1)
    assert fc_1d > 0.0

    # Path forecast
    path = model.forecast_path(horizon=5)
    assert len(path) == 5
    assert (path > 0.0).all()


def test_egarch_model(returns_sample):
    ret, _ = returns_sample
    model = EGARCHModel(p=1, q=1)
    res = model.fit(ret)

    assert res.log_likelihood is not None
    assert isinstance(model.leverage_coefficient, float)
    fc = model.forecast(horizon=1)
    assert fc > 0.0


def test_gjr_garch_model(returns_sample):
    ret, _ = returns_sample
    model = GJRGARCHModel(p=1, o=1, q=1)
    res = model.fit(ret)

    assert res.bic is not None
    assert isinstance(model.asymmetry_coefficient, float)
    fc = model.forecast(horizon=1)
    assert fc > 0.0


def test_har_rv_model(returns_sample):
    _, true_var = returns_sample
    har = HARRVModel()
    har.fit(true_var, forecast_horizon=1)

    assert har.r_squared_ > 0.0
    fc = har.forecast(true_var)
    assert fc > 0.0

    fc_vol = har.forecast_volatility_annualized(true_var)
    assert fc_vol > 0.0
