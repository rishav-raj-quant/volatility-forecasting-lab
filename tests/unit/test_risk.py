"""Unit tests for quantitative risk modeling (VaR, Expected Shortfall, Backtesting)."""

import pytest
import numpy as np
import pandas as pd
from volatility_lab.risk.var import ValueAtRisk
from volatility_lab.risk.expected_shortfall import ExpectedShortfall
from volatility_lab.risk.backtesting import VaRBacktester


@pytest.fixture
def returns_data():
    dates = pd.date_range("2021-01-01", periods=600, freq="B")
    np.random.seed(42)
    # Student-t returns with fat tails (df=4)
    r = np.random.standard_t(df=4, size=600) * 0.01
    return pd.Series(r, index=dates)


def test_value_at_risk_hierarchy(returns_data):
    var_calc = ValueAtRisk(confidence_levels=[0.90, 0.95, 0.99])
    df_var = var_calc.parametric_normal_var(returns_data)

    valid = df_var.dropna()
    # Higher confidence level must yield strictly higher loss threshold
    assert (valid["var_norm_99"] > valid["var_norm_95"]).all()
    assert (valid["var_norm_95"] > valid["var_norm_90"]).all()


def test_expected_shortfall_coherence(returns_data):
    var_calc = ValueAtRisk(confidence_levels=[0.95])
    df_var = var_calc.parametric_normal_var(returns_data)

    es_calc = ExpectedShortfall(confidence_levels=[0.95])
    df_es = es_calc.parametric_normal_es(returns_data)

    valid_var = df_var["var_norm_95"].dropna()
    valid_es = df_es["es_norm_95"].dropna()
    common_idx = valid_var.index.intersection(valid_es.index)

    # Subadditivity & coherence: Expected Shortfall must be >= VaR
    assert (valid_es.loc[common_idx] >= valid_var.loc[common_idx]).all()


def test_var_backtest_kupiec_and_christoffersen(returns_data):
    var_calc = ValueAtRisk(confidence_levels=[0.95])
    var_95 = var_calc.parametric_normal_var(returns_data)["var_norm_95"].dropna()
    matched_ret = returns_data.reindex(var_95.index)

    res = VaRBacktester.backtest(matched_ret, var_95, confidence_level=0.95)

    assert res.total_observations == len(matched_ret)
    assert res.actual_violations > 0
    assert res.kupiec_p_value >= 0.0
    assert res.christoffersen_ind_p_value >= 0.0
    assert res.conditional_coverage_p_value >= 0.0
