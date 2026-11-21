"""Loss functions and evaluation metrics for volatility forecasting."""

from typing import Dict, Union
import numpy as np
import pandas as pd


def mse(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Mean Squared Error: (1 / N) * sum((y_true - y_pred)^2)."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Root Mean Squared Error: sqrt(MSE)."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Mean Absolute Error: (1 / N) * sum(|y_true - y_pred|)."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def qlike_loss(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    eps: float = 1e-8,
) -> float:
    """Patton (2011) Quasi-Likelihood (QLIKE) robust loss function.

    QLIKE(h_t, sigma^2_t) = (sigma^2_t / h_t) - ln(sigma^2_t / h_t) - 1
    where sigma^2_t is the realized variance proxy (y_true) and h_t is the model forecast (y_pred).
    QLIKE is strictly robust to noisy volatility proxies (unlike symmetric MSE).
    """
    yt = np.maximum(np.asarray(y_true, dtype=float), eps)
    yp = np.maximum(np.asarray(y_pred, dtype=float), eps)

    ratio = yt / yp
    loss = ratio - np.log(ratio) - 1.0
    return float(np.mean(loss))


def mape(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    eps: float = 1e-8,
) -> float:
    """Mean Absolute Percentage Error in percent."""
    yt = np.maximum(np.asarray(y_true, dtype=float), eps)
    yp = np.maximum(np.asarray(y_pred, dtype=float), eps)
    return float(np.mean(np.abs((yt - yp) / yt)) * 100.0)


def r2_score(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Coefficient of Determination R^2."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_all_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Computes all core research metrics for a prediction vector."""
    return {
        "qlike": qlike_loss(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
    }
