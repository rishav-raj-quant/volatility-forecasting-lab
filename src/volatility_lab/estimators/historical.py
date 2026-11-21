"""Historical Close-to-Close volatility estimator."""

from typing import List, Union
import numpy as np
import pandas as pd
from volatility_lab.utils.calendar import SQRT_252


class HistoricalVolatilityEstimator:
    """Computes rolling sample standard deviation of log returns across specified windows."""

    def __init__(self, windows: Union[int, List[int]] = 20, annualize: bool = True):
        self.windows = [windows] if isinstance(windows, int) else windows
        self.annualize = annualize

    def estimate(self, returns: pd.Series) -> pd.DataFrame:
        """Estimates historical volatility for each rolling window.

        Args:
            returns: Daily log returns series.

        Returns:
            DataFrame with historical volatility for each window.
        """
        results = {}
        factor = SQRT_252 if self.annualize else 1.0

        for w in self.windows:
            col_name = f"hist_vol_{w}d"
            # Rolling sample standard deviation (ddof=1)
            vol = returns.rolling(window=w, min_periods=max(2, w // 2)).std() * factor
            results[col_name] = vol

        return pd.DataFrame(results, index=returns.index)
