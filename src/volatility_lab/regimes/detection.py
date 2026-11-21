"""Volatility regime classification: Statistical thresholds, GMM, and Markov regimes."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


class StatisticalRegimeDetector:
    """Classifies volatility into 4 regimes based strictly on historical expanding percentiles:
    0: Low Volatility (0% - 25%)
    1: Normal Volatility (25% - 75%)
    2: High Volatility (75% - 95%)
    3: Extreme Volatility (> 95%)
    """

    def __init__(self, warmup_period: int = 252):
        self.warmup_period = warmup_period
        self.regime_names = {
            0: "Low Volatility",
            1: "Normal Volatility",
            2: "High Volatility",
            3: "Extreme Volatility",
        }

    def detect(self, volatility_series: pd.Series) -> pd.Series:
        """Classifies each time step using only historical expanding distribution to prevent lookahead."""
        vol = volatility_series.dropna()
        n = len(vol)
        regimes = np.zeros(n, dtype=int)

        # Warmup period classification using initial sample
        warmup = min(self.warmup_period, n)
        p25_init = np.percentile(vol.iloc[:warmup], 25)
        p75_init = np.percentile(vol.iloc[:warmup], 75)
        p95_init = np.percentile(vol.iloc[:warmup], 95)

        for i in range(warmup):
            v = vol.iloc[i]
            if v <= p25_init:
                regimes[i] = 0
            elif v <= p75_init:
                regimes[i] = 1
            elif v <= p95_init:
                regimes[i] = 2
            else:
                regimes[i] = 3

        # Expanding window thresholds for remaining observations
        for i in range(warmup, n):
            hist = vol.iloc[:i]
            p25 = np.percentile(hist, 25)
            p75 = np.percentile(hist, 75)
            p95 = np.percentile(hist, 95)
            v = vol.iloc[i]

            if v <= p25:
                regimes[i] = 0
            elif v <= p75:
                regimes[i] = 1
            elif v <= p95:
                regimes[i] = 2
            else:
                regimes[i] = 3

        return pd.Series(regimes, index=vol.index, name="regime_code")


class GMMRegimeDetector:
    """Gaussian Mixture Model regime detector for latent state identification."""

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.gmm = GaussianMixture(n_components=n_regimes, random_state=random_state, max_iter=200)
        self.state_map_: Dict[int, int] = {}

    def fit_predict(self, volatility_series: pd.Series) -> pd.Series:
        """Fits GMM and orders regimes by increasing average volatility level."""
        clean = volatility_series.dropna()
        X = clean.values.reshape(-1, 1)
        raw_states = self.gmm.fit_predict(X)

        # Order clusters by mean volatility: 0 = Lowest, (n_regimes-1) = Highest
        means = self.gmm.means_.flatten()
        sorted_indices = np.argsort(means)
        self.state_map_ = {orig: ranked for ranked, orig in enumerate(sorted_indices)}

        ranked_states = np.array([self.state_map_[s] for s in raw_states])
        return pd.Series(ranked_states, index=clean.index, name="gmm_regime")
