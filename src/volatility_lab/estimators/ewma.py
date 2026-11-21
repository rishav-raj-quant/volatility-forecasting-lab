"""Exponentially Weighted Moving Average (EWMA) volatility estimator."""

from typing import Union
import numpy as np
import pandas as pd
from volatility_lab.utils.calendar import SQRT_252


class EWMAVolatilityEstimator:
    """RiskMetrics Exponentially Weighted Moving Average Volatility Estimator.

    sigma^2_t = lambda * sigma^2_{t-1} + (1 - lambda) * r^2_{t-1}
    """

    def __init__(self, decay_factor: float = 0.94, annualize: bool = True):
        if not (0.0 < decay_factor < 1.0):
            raise ValueError(f"Decay factor lambda must be in (0, 1), received {decay_factor}")
        self.decay_factor = decay_factor
        self.annualize = annualize

    def estimate(self, returns: pd.Series) -> pd.Series:
        """Computes recursive EWMA conditional variance and volatility series.

        Args:
            returns: Daily log returns series.

        Returns:
            Series of EWMA volatility.
        """
        clean_ret = returns.dropna()
        n = len(clean_ret)
        if n == 0:
            return pd.Series(dtype=float, index=returns.index)

        variances = np.zeros(n)
        # Seed initial variance with unconditional sample variance of first 20 observations or entire sample
        seed_len = min(20, n)
        variances[0] = np.var(clean_ret.iloc[:seed_len].values, ddof=1) if seed_len > 1 else clean_ret.iloc[0]**2
        if variances[0] <= 0:
            variances[0] = 1e-4

        lam = self.decay_factor
        one_minus_lam = 1.0 - lam
        ret_sq = clean_ret.values ** 2

        for t in range(1, n):
            variances[t] = lam * variances[t - 1] + one_minus_lam * ret_sq[t - 1]

        volatility = np.sqrt(variances)
        if self.annualize:
            volatility *= SQRT_252

        return pd.Series(volatility, index=clean_ret.index, name=f"ewma_vol_lambda_{self.decay_factor}")
