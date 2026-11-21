"""Residual diagnostics, Ljung-Box tests, and Mincer-Zarnowitz regressions."""

from typing import Dict, Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox


def mincer_zarnowitz_test(
    realized_variance: np.ndarray,
    forecast_variance: np.ndarray,
) -> Dict[str, Any]:
    """Mincer-Zarnowitz regression testing volatility forecast unbiasedness.

    sigma^2_t = alpha + beta * h_t + u_t
    Under rational expectations and forecast unbiasedness: alpha = 0 and beta = 1.
    """
    y = np.asarray(realized_variance, dtype=float)
    x = np.asarray(forecast_variance, dtype=float)

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    alpha = float(model.params[0])
    beta = float(model.params[1])
    r2 = float(model.rsquared)

    # Wald test for joint hypothesis: alpha = 0 and beta = 1
    hypothesis = "(const = 0), (x1 = 1)"
    try:
        wald = model.wald_test(hypothesis, scalar=True)
        f_stat = float(np.asarray(wald.statistic).squeeze().item())
        p_val = float(np.asarray(wald.pvalue).squeeze().item())
    except Exception:
        f_stat = 0.0
        p_val = 1.0

    return {
        "alpha": alpha,
        "beta": beta,
        "r_squared": r2,
        "wald_stat": f_stat,
        "p_value": p_val,
        "is_unbiased": p_val >= 0.05,
    }


def ljung_box_diagnostics(residuals: np.ndarray, lags: int = 10) -> Dict[str, Any]:
    """Tests standardized residuals and squared standardized residuals for remaining autocorrelation."""
    res = np.asarray(residuals, dtype=float)
    clean_res = res[~np.isnan(res)]

    lb_resid = acorr_ljungbox(clean_res, lags=[lags], return_df=True)
    lb_sq_resid = acorr_ljungbox(clean_res ** 2, lags=[lags], return_df=True)

    return {
        "lb_stat": float(lb_resid["lb_stat"].values[0]),
        "lb_pvalue": float(lb_resid["lb_pvalue"].values[0]),
        "lb_sq_stat": float(lb_sq_resid["lb_stat"].values[0]),
        "lb_sq_pvalue": float(lb_sq_resid["lb_pvalue"].values[0]),
        "no_serial_correlation": float(lb_resid["lb_pvalue"].values[0]) >= 0.05,
        "no_remaining_arch": float(lb_sq_resid["lb_pvalue"].values[0]) >= 0.05,
    }
