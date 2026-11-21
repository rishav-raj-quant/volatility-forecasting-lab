"""Glosten-Jagannathan-Runkle (GJR-GARCH) model for threshold asymmetry."""

from typing import Optional
import numpy as np
import pandas as pd
from arch import arch_model as _arch_model
from volatility_lab.econometrics.arch_model import EconometricModelResult
from volatility_lab.utils.calendar import SQRT_252
from volatility_lab.utils.logging import get_logger

logger = get_logger(__name__)


class GJRGARCHModel:
    """GJR-GARCH(p, o, q) model (Glosten, Jagannathan, Runkle 1993).

    sigma^2_t = omega + (alpha + gamma * I_{t-1}) * epsilon^2_{t-1} + beta * sigma^2_{t-1}
    where I_{t-1} = 1 if epsilon_{t-1} < 0, else 0.
    Positive gamma indicates that bad news (negative returns) generates higher volatility than good news.
    """

    def __init__(self, p: int = 1, o: int = 1, q: int = 1, dist: str = "normal"):
        self.p = p
        self.o = o
        self.q = q
        self.dist = dist
        self.fitted_model_ = None
        self.last_result_: Optional[EconometricModelResult] = None

    def fit(self, returns: pd.Series) -> "GJRGARCHModel":
        """Fits GJR-GARCH model using maximum likelihood."""
        clean = returns.dropna() * 100.0
        am = _arch_model(clean, mean="Constant", vol="Garch", p=self.p, o=self.o, q=self.q, dist=self.dist)
        res = am.fit(disp="off")
        self.fitted_model_ = res

        cond_vol = (res.conditional_volatility / 100.0) * SQRT_252
        std_resid = res.std_resid

        result = EconometricModelResult(
            model_name=f"GJR-GARCH({self.p},{self.o},{self.q})-{self.dist}",
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
    def asymmetry_coefficient(self) -> float:
        """Threshold asymmetry parameter gamma. Value > 0 confirms leverage effect."""
        if self.last_result_ is None:
            return 0.0
        return float(self.last_result_.params.get("gamma[1]", 0.0))

    def forecast(self, horizon: int = 1) -> float:
        """Forecasts annualized conditional volatility over horizon steps."""
        if self.fitted_model_ is None:
            raise RuntimeError("Model must be fitted before forecasting.")
        fc = self.fitted_model_.forecast(horizon=horizon)
        var_h = fc.variance.values[-1, -1]
        vol_daily = np.sqrt(var_h) / 100.0
        return float(vol_daily * SQRT_252)
