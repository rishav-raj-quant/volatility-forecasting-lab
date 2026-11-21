"""Value at Risk (VaR) estimation across parametric, historical, and econometric methods."""

from typing import Dict, Union, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


class ValueAtRisk:
    """Calculates Value at Risk (VaR) for loss distributions.

    Note: VaR is a threshold quantile representing the minimum loss expected over a horizon
    at confidence level alpha. It is NOT the maximum possible loss.
    """

    def __init__(self, confidence_levels: List[float] = [0.90, 0.95, 0.99]):
        self.confidence_levels = confidence_levels

    def historical_var(self, returns: pd.Series, window: int = 252) -> pd.DataFrame:
        """Rolling historical empirical quantile VaR on negative return (loss) space."""
        r = returns.dropna()
        results = {}
        for cl in self.confidence_levels:
            alpha = 1.0 - cl
            # Loss = -return, so VaR_alpha is positive loss threshold
            var_series = (-r).rolling(window=window, min_periods=window // 2).quantile(cl)
            results[f"var_hist_{int(cl*100)}"] = var_series
        return pd.DataFrame(results, index=r.index)

    def parametric_normal_var(
        self,
        returns: pd.Series,
        window: int = 252,
    ) -> pd.DataFrame:
        """Rolling Gaussian parametric VaR: mu_loss + z_alpha * sigma."""
        r = returns.dropna()
        mu = (-r).rolling(window=window).mean()
        sigma = r.rolling(window=window).std()

        results = {}
        for cl in self.confidence_levels:
            z_score = stats.norm.ppf(cl)
            results[f"var_norm_{int(cl*100)}"] = mu + z_score * sigma
        return pd.DataFrame(results, index=r.index)

    def parametric_student_t_var(
        self,
        returns: pd.Series,
        window: int = 252,
        degrees_of_freedom: int = 5,
    ) -> pd.DataFrame:
        """Rolling Student's t parametric VaR to account for fat tails and leptokurtosis."""
        r = returns.dropna()
        mu = (-r).rolling(window=window).mean()
        sigma = r.rolling(window=window).std()

        # Adjust scale for t-distribution variance = df / (df - 2)
        scale = sigma * np.sqrt((degrees_of_freedom - 2) / degrees_of_freedom)

        results = {}
        for cl in self.confidence_levels:
            t_crit = stats.t.ppf(cl, df=degrees_of_freedom)
            results[f"var_t_{int(cl*100)}"] = mu + t_crit * scale
        return pd.DataFrame(results, index=r.index)

    def conditional_volatility_var(
        self,
        conditional_daily_volatility: pd.Series,
        mean_return: float = 0.0,
        dist: str = "normal",
        df: int = 5,
    ) -> pd.DataFrame:
        """Dynamic VaR driven by conditional volatility forecasts (EWMA or GARCH)."""
        vol = conditional_daily_volatility.dropna()
        results = {}
        for cl in self.confidence_levels:
            if dist == "t":
                crit = stats.t.ppf(cl, df=df) * np.sqrt((df - 2) / df)
            else:
                crit = stats.norm.ppf(cl)
            results[f"var_dyn_{int(cl*100)}"] = -mean_return + crit * vol
        return pd.DataFrame(results, index=vol.index)
