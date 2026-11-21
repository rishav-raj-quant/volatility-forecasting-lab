"""Unit tests for InstitutionalSignalEngine."""

import pytest
import pandas as pd
from volatility_lab.forecasting.signals import InstitutionalSignalEngine


def test_buy_signal_evaluation():
    prev_row = pd.Series({
        "Close": 100.0,
        "sma_9": 99.5,
        "log_return": 0.002,
        "ewma_vol_ann": 14.0,
        "garman_klass_ann": 13.5,
        "jump_component": 0.05,
        "var_95_pct": 2.0,
        "volatility_regime": 1,
    })
    curr_row = pd.Series({
        "Close": 101.5,
        "sma_9": 99.8,
        "log_return": 0.015,
        "ewma_vol_ann": 14.2,
        "garman_klass_ann": 15.0,  # Range expanding
        "jump_component": 0.05,
        "var_95_pct": 2.1,
        "volatility_regime": 1,
    })

    sig = InstitutionalSignalEngine.evaluate(curr_row, prev_row)
    assert sig["signal_type"] == "BUY"
    assert sig["color"] == "dark_green"
    assert sig["confidence_score"] >= 80.0
    assert sig["stop_loss"] < 101.5
    assert sig["target_price"] > 101.5


def test_sell_signal_evaluation():
    prev_row = pd.Series({
        "Close": 100.0,
        "sma_9": 100.0,
        "log_return": 0.0,
        "ewma_vol_ann": 15.0,
        "garman_klass_ann": 15.0,
        "jump_component": 0.05,
        "var_95_pct": 2.0,
        "volatility_regime": 1,
    })
    curr_row = pd.Series({
        "Close": 95.0,
        "sma_9": 99.0,
        "log_return": -0.05,  # Extreme breach
        "ewma_vol_ann": 35.0,
        "garman_klass_ann": 32.0,
        "jump_component": 0.40,  # Extreme jump shock
        "var_95_pct": 4.5,
        "volatility_regime": 3,
    })

    sig = InstitutionalSignalEngine.evaluate(curr_row, prev_row)
    assert sig["signal_type"] == "SELL"
    assert sig["color"] == "dark_red"
    assert sig["confidence_score"] >= 80.0
    assert "LIQUIDATE" in sig["action"]


def test_neutral_signal_evaluation():
    row = pd.Series({
        "Close": 100.0,
        "sma_9": 100.0,
        "log_return": 0.0,
        "ewma_vol_ann": 16.0,
        "garman_klass_ann": 16.0,
        "jump_component": 0.05,
        "var_95_pct": 2.0,
        "volatility_regime": 1,
    })
    sig = InstitutionalSignalEngine.evaluate(row, row)
    assert sig["signal_type"] == "NEUTRAL"
    assert sig["color"] == "dark_slate"
