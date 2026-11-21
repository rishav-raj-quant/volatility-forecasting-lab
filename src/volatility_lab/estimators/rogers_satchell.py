"""Rogers-Satchell (1991) drift-independent range volatility estimator."""

import numpy as np
import pandas as pd
from volatility_lab.utils.calendar import SQRT_252


class RogersSatchellEstimator:
    """Rogers-Satchell (1991) Volatility Estimator.

    sigma^2 = (1 / N) * sum( ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O) )
    Maintains unbiased estimation even when underlying drift mu != 0.
    """

    def __init__(self, window: int = 20, annualize: bool = True):
        self.window = window
        self.annualize = annualize

    def estimate(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Computes rolling Rogers-Satchell volatility."""
        h = np.maximum(high.values, 1e-4)
        l = np.maximum(low.values, 1e-4)
        o = np.maximum(open_.values, 1e-4)
        c = np.maximum(close.values, 1e-4)

        log_hc = np.log(h / c)
        log_ho = np.log(h / o)
        log_lc = np.log(l / c)
        log_lo = np.log(l / o)

        rs_daily = log_hc * log_ho + log_lc * log_lo
        rs_series = pd.Series(rs_daily, index=close.index)

        rolling_var = rs_series.rolling(window=self.window, min_periods=max(2, self.window // 2)).mean()
        vol = np.sqrt(np.maximum(rolling_var, 0.0))

        if self.annualize:
            vol *= SQRT_252

        return pd.Series(vol, index=close.index, name=f"rogers_satchell_vol_{self.window}d")
