"""Garman-Klass (1980) OHLC volatility estimator."""

import numpy as np
import pandas as pd
from volatility_lab.utils.calendar import SQRT_252


class GarmanKlassEstimator:
    """Garman-Klass (1980) Volatility Estimator.

    sigma^2 = (1 / N) * sum( 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2 )
    Offers up to 8x efficiency improvement over close-to-close estimator.
    """

    def __init__(self, window: int = 20, annualize: bool = True):
        self.window = window
        self.annualize = annualize
        self.c_term = 2.0 * np.log(2.0) - 1.0

    def estimate(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Computes rolling Garman-Klass volatility from OHLC series."""
        log_hl = np.log(np.maximum(high.values, 1e-4) / np.maximum(low.values, 1e-4))
        log_co = np.log(np.maximum(close.values, 1e-4) / np.maximum(open_.values, 1e-4))

        gk_daily_var = 0.5 * (log_hl ** 2) - self.c_term * (log_co ** 2)
        gk_series = pd.Series(gk_daily_var, index=close.index)

        rolling_var = gk_series.rolling(window=self.window, min_periods=max(2, self.window // 2)).mean()
        vol = np.sqrt(np.maximum(rolling_var, 0.0))

        if self.annualize:
            vol *= SQRT_252

        return pd.Series(vol, index=close.index, name=f"garman_klass_vol_{self.window}d")
