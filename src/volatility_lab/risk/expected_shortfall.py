"""Expected Shortfall (Conditional Value at Risk / CVaR) estimation."""

from typing import Dict, List, Union
import numpy as np
import pandas as pd
from scipy import stats


class ExpectedShortfall:
    """Expected Shortfall (CVaR) - coherent risk measure calculating average loss in the tail beyond VaR.

    ES_alpha = E[ L | L > VaR_alpha ]
    """

    def __init__(self, confidence_levels: List[float] = [0.90, 0.95, 0.99]):
        self.confidence_levels = confidence_levels

    def historical_es(self, returns: pd.Series, window: int = 252) -> pd.DataFrame:
        """Rolling historical Expected Shortfall."""
        r = returns.dropna()
        losses = -r
        results = {}

        for cl in self.confidence_levels:
            es_vals = []
            for i in range(len(losses)):
                if i < window:
                    es_vals.append(np.nan)
                    continue
                w_losses = losses.iloc[i - window : i]
                var_threshold = np.percentile(w_losses, cl * 100.0)
                tail_losses = w_losses[w_losses >= var_threshold]
                es = tail_losses.mean() if len(tail_losses) > 0 else var_threshold
                es_vals.append(es)
            results[f"es_hist_{int(cl*100)}"] = pd.Series(es_vals, index=losses.index)

        return pd.DataFrame(results, index=losses.index)

    def parametric_normal_es(
        self,
        returns: pd.Series,
        window: int = 252,
    ) -> pd.DataFrame:
        """Gaussian parametric Expected Shortfall: mu_loss + sigma * (phi(z_alpha) / (1 - alpha))."""
        r = returns.dropna()
        mu = (-r).rolling(window=window).mean()
        sigma = r.rolling(window=window).std()

        results = {}
        for cl in self.confidence_levels:
            alpha = 1.0 - cl
            z_crit = stats.norm.ppf(cl)
            pdf_z = stats.norm.pdf(z_crit)
            results[f"es_norm_{int(cl*100)}"] = mu + sigma * (pdf_z / alpha)

        return pd.DataFrame(results, index=r.index)
