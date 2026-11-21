"""Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model."""

from typing import Optional, Dict
import numpy as np
import pandas as pd
from arch import arch_model as _arch_model
from volatility_lab.econometrics.arch_model import EconometricModelResult
from volatility_lab.utils.calendar import SQRT_252
from volatility_lab.utils.logging import get_logger

logger = get_logger(__name__)


class GARCHModel:
    """GARCH(p, q) model. Default is standard benchmark GARCH(1,1).

    sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}
    Stationarity requires alpha + beta < 1.
    """

    def __init__(self, p: int = 1, q: int = 1, dist: str = "normal"):
        self.p = p
        self.q = q
        self.dist = dist
        self.fitted_model_ = None
        self.last_result_: Optional[EconometricModelResult] = None

    def fit(self, returns: pd.Series) -> "GARCHModel":
        """Fits GARCH(p,q) model using quasi-maximum likelihood estimation."""
        clean = returns.dropna() * 100.0
        am = _arch_model(clean, mean="Constant", vol="Garch", p=self.p, q=self.q, dist=self.dist)
        res = am.fit(disp="off")
        self.fitted_model_ = res

        cond_vol = (res.conditional_volatility / 100.0) * SQRT_252
        std_resid = res.std_resid

        result = EconometricModelResult(
            model_name=f"GARCH({self.p},{self.q})-{self.dist}",
            params=dict(res.params),
            aic=float(res.aic),
            bic=float(res.bic),
            log_likelihood=float(res.loglikelihood),
            conditional_volatility=cond_vol,
            residuals=res.resid / 100.0,
            standardized_residuals=std_resid,
        )
        self.last_result_ = result
        return self

    @property
    def aic(self) -> float:
        return self.last_result_.aic if self.last_result_ else 0.0

    @property
    def bic(self) -> float:
        return self.last_result_.bic if self.last_result_ else 0.0

    @property
    def log_likelihood(self) -> float:
        return self.last_result_.log_likelihood if self.last_result_ else 0.0

    @property
    def persistence(self) -> float:
        """Computes volatility persistence sum(alpha_i) + sum(beta_j)."""
        if self.last_result_ is None:
            return 0.0
        params = self.last_result_.params
        alpha = sum(v for k, v in params.items() if k.startswith("alpha"))
        beta = sum(v for k, v in params.items() if k.startswith("beta"))
        return float(alpha + beta)

    @property
    def half_life(self) -> float:
        """Half-life of volatility shocks in trading days: ln(0.5) / ln(persistence)."""
        pers = self.persistence
        if 0.0 < pers < 1.0:
            return float(np.log(0.5) / np.log(pers))
        return float("inf")

    def forecast(self, horizon: int = 1) -> float:
        """Forecasts annualized conditional volatility over horizon steps."""
        if self.fitted_model_ is None:
            raise RuntimeError("Model must be fitted before forecasting.")
        fc = self.fitted_model_.forecast(horizon=horizon)
        var_h = fc.variance.values[-1, -1]
        vol_daily = np.sqrt(var_h) / 100.0
        return float(vol_daily * SQRT_252)

    def forecast_path(self, horizon: int = 20) -> pd.Series:
        """Forecasts multi-step annualized volatility trajectory from h=1 to horizon."""
        if self.fitted_model_ is None:
            raise RuntimeError("Model must be fitted before forecasting.")
        fc = self.fitted_model_.forecast(horizon=horizon)
        variances = fc.variance.values[-1, :]
        vol_path = (np.sqrt(variances) / 100.0) * SQRT_252
        return pd.Series(vol_path, index=range(1, horizon + 1), name="garch_vol_forecast_path")
