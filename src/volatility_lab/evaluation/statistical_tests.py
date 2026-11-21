"""Diebold-Mariano test with Newey-West long-run variance adjustment."""

from dataclasses import dataclass
from typing import Union, Literal
import numpy as np
from scipy import stats


@dataclass
class DieboldMarianoResult:
    """Outcome of Diebold-Mariano predictive accuracy comparison."""
    statistic: float
    p_value: float
    loss_function: str
    model1_mean_loss: float
    model2_mean_loss: float
    superior_model: str


def diebold_mariano_test(
    y_true: Union[np.ndarray, list],
    y_pred1: Union[np.ndarray, list],
    y_pred2: Union[np.ndarray, list],
    horizon: int = 1,
    loss_type: Literal["qlike", "mse", "mae"] = "qlike",
) -> DieboldMarianoResult:
    """Executes Diebold-Mariano test comparing forecast accuracy between model 1 and model 2.

    H0: E[d_t] = 0 (Both models have identical predictive accuracy)
    H1: E[d_t] != 0 (One model significantly outperforms)

    Adjusts variance for horizon-dependent serial correlation via Newey-West kernel.
    """
    yt = np.asarray(y_true, dtype=float)
    yp1 = np.asarray(y_pred1, dtype=float)
    yp2 = np.asarray(y_pred2, dtype=float)

    if loss_type == "qlike":
        # QLIKE: yt/yp - ln(yt/yp) - 1
        eps = 1e-8
        yt_safe = np.maximum(yt, eps)
        yp1_safe = np.maximum(yp1, eps)
        yp2_safe = np.maximum(yp2, eps)
        l1 = (yt_safe / yp1_safe) - np.log(yt_safe / yp1_safe) - 1.0
        l2 = (yt_safe / yp2_safe) - np.log(yt_safe / yp2_safe) - 1.0
    elif loss_type == "mse":
        l1 = (yt - yp1) ** 2
        l2 = (yt - yp2) ** 2
    elif loss_type == "mae":
        l1 = np.abs(yt - yp1)
        l2 = np.abs(yt - yp2)
    else:
        raise ValueError(f"Unsupported loss_type '{loss_type}'. Must be 'qlike', 'mse', or 'mae'.")

    # Differential loss series: d_t = l1_t - l2_t
    d = l1 - l2
    t = len(d)
    if t < 5:
        return DieboldMarianoResult(
            statistic=0.0,
            p_value=1.0,
            loss_function=loss_type,
            model1_mean_loss=float(np.mean(l1)),
            model2_mean_loss=float(np.mean(l2)),
            superior_model="Insufficient Data",
        )

    mean_d = np.mean(d)

    # Newey-West spectral variance at zero frequency (lag = horizon - 1, min 1)
    max_lag = max(1, horizon - 1)
    gamma_0 = np.var(d, ddof=1)

    gamma_sum = 0.0
    for k in range(1, max_lag + 1):
        # Sample autocovariance at lag k
        gamma_k = np.mean((d[k:] - mean_d) * (d[:-k] - mean_d))
        weight = 1.0 - (k / (max_lag + 1.0))  # Bartlett kernel
        gamma_sum += 2.0 * weight * gamma_k

    lr_var = gamma_0 + gamma_sum
    if lr_var <= 0:
        lr_var = gamma_0

    dm_stat = mean_d / np.sqrt(lr_var / t)
    # Two-sided p-value
    p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))

    if p_val < 0.05:
        superior = "Model 1" if mean_d < 0 else "Model 2"
    else:
        superior = "Indistinguishable (p >= 0.05)"

    return DieboldMarianoResult(
        statistic=float(dm_stat),
        p_value=float(p_val),
        loss_function=loss_type,
        model1_mean_loss=float(np.mean(l1)),
        model2_mean_loss=float(np.mean(l2)),
        superior_model=superior,
    )
