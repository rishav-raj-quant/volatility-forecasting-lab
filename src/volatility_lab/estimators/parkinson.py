"""Parkinson (1980) extreme-value High-Low range volatility estimator."""

import numpy as np
import pandas as pd
from volatility_lab.utils.calendar import SQRT_252


class ParkinsonEstimator:
    """Parkinson (1980) Volatility Estimator.

    sigma^2 = (1 / (4 * ln(2))) * (1 / N) * sum( (ln(H_i / L_i))^2 )
    Approximately 5x more efficient than standard close-to-close estimator.
    """

    def __init__(self, window: int = 20, annualize: bool = True):
        self.window = window
        self.annualize = annualize
        self.scale = 1.0 / (4.0 * np.log(2.0))

    def estimate(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """Computes rolling Parkinson volatility."""
        hl_ratio = np.log(np.maximum(high.values, 1e-4) / np.maximum(low.values, 1e-4))
        hl_sq = pd.Series(hl_ratio ** 2, index=high.index)

        rolling_var = self.scale * hl_sq.rolling(window=self.window, min_periods=max(2, self.window // 2)).mean()
        vol = np.sqrt(np.maximum(rolling_var, 0.0))

        if self.annualize:
            vol *= SQRT_252

        return pd.Series(vol, index=high.index, name=f"parkinson_vol_{self.window}d")
