"""Unit tests for evaluation metrics, statistical tests, and walk-forward validation."""

import pytest
import numpy as np
import pandas as pd
from volatility_lab.evaluation.metrics import (
    mse,
    rmse,
    mae,
    qlike_loss,
    r2_score,
    compute_all_metrics,
)
from volatility_lab.evaluation.statistical_tests import diebold_mariano_test
from volatility_lab.evaluation.walk_forward import WalkForwardValidator
from volatility_lab.evaluation.diagnostics import mincer_zarnowitz_test


def test_loss_metrics():
    y_true = np.array([0.04, 0.09, 0.01, 0.16])
    y_pred = np.array([0.05, 0.08, 0.02, 0.15])

    val_mse = mse(y_true, y_pred)
    val_rmse = rmse(y_true, y_pred)
    val_mae = mae(y_true, y_pred)
    val_qlike = qlike_loss(y_true, y_pred)

    assert np.isclose(val_rmse, np.sqrt(val_mse))
    assert val_mae > 0.0
    assert val_qlike > 0.0


def test_qlike_zero_error():
    # If prediction matches true realized variance perfectly, QLIKE loss = 0.0
    # ratio = 1, ln(1) = 0 => 1 - 0 - 1 = 0
    y = np.array([0.04, 0.09, 0.16])
    assert np.isclose(qlike_loss(y, y), 0.0)


def test_diebold_mariano_test():
    np.random.seed(42)
    y_true = np.random.exponential(0.04, 100)
    # Model 1 has small error, Model 2 has huge error
    y_pred1 = y_true + np.random.normal(0, 0.005, 100)
    y_pred2 = y_true + np.random.normal(0, 0.05, 100)

    res = diebold_mariano_test(y_true, y_pred1, y_pred2, horizon=1, loss_type="qlike")
    assert res.statistic < 0.0  # Model 1 has lower loss
    assert res.p_value < 0.05
    assert res.superior_model == "Model 1"


def test_walk_forward_splits():
    validator = WalkForwardValidator(min_train_size=100, test_size=20, step_size=20)
    n = 200
    splits = list(validator.split(n, forecast_horizon=5))

    assert len(splits) > 0
    for train_idx, test_idx in splits:
        # Strict time-ordering: all train timestamps precede test timestamps
        assert train_idx[-1] < test_idx[0]
        # Purging check: test_idx[0] - train_idx[-1] must be >= forecast_horizon
        assert test_idx[0] - train_idx[-1] >= 5


def test_mincer_zarnowitz():
    np.random.seed(42)
    y_true = np.random.exponential(0.04, 100)
    y_pred = y_true + np.random.normal(0, 0.002, 100)

    res = mincer_zarnowitz_test(y_true, y_pred)
    assert "alpha" in res
    assert "beta" in res
    assert "p_value" in res
    assert res["r_squared"] > 0.8
