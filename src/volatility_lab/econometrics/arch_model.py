"""Autoregressive Conditional Heteroskedasticity (ARCH) model."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from arch import arch_model as _arch_model
from volatility_lab.utils.calendar import SQRT_252
from volatility_lab.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EconometricModelResult:
    """Standardized output container for econometric volatility models."""
    model_name: str
    params: Dict[str, float]
    aic: float
    bic: float
    log_likelihood: float
    conditional_volatility: pd.Series
    residuals: pd.Series
    standardized_residuals: pd.Series


class ARCHModel:
    """ARCH(p) volatility model using maximum likelihood estimation."""

    def __init__(self, p: int = 1, mean: str = "Constant", dist: str = "normal"):
        self.p = p
        self.mean = mean
        self.dist = dist
        self.fitted_model_ = None
        self.last_result_: Optional[EconometricModelResult] = None

    def fit(self, returns: pd.Series) -> EconometricModelResult:
        """Fits ARCH(p) model via Maximum Likelihood Estimation."""
        clean = returns.dropna() * 100.0  # Scale returns by 100 for stable optimization in arch package
        am = _arch_model(clean, mean=self.mean, vol="ARCH", p=self.p, dist=self.dist)
        res = am.fit(disp="off")
        self.fitted_model_ = res

        # Rescale conditional volatility back to original decimal units
        cond_vol = (res.conditional_volatility / 100.0) * SQRT_252
        std_resid = res.std_resid

        result = EconometricModelResult(
            model_name=f"ARCH({self.p})",
            params=dict(res.params),
            aic=float(res.aic),
            bic=float(res.bic),
            log_likelihood=float(res.loglikelihood),
            conditional_volatility=cond_vol,
            residuals=res.resid / 100.0,
            standardized_residuals=std_resid,
        )
        self.last_result_ = result
        return result

    def forecast(self, horizon: int = 1) -> float:
        """Forecasts annualized conditional volatility over horizon steps."""
        if self.fitted_model_ is None:
            raise RuntimeError("Model must be fitted before forecasting.")
        fc = self.fitted_model_.forecast(horizon=horizon)
        # Final forecast variance
        var_h = fc.variance.values[-1, -1]  # scaled by 100^2
        vol_daily = np.sqrt(var_h) / 100.0
        return float(vol_daily * SQRT_252)
