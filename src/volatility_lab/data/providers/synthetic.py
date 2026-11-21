"""Synthetic market data generator with realistic stochastic volatility dynamics."""

import numpy as np
import pandas as pd
from typing import Optional
from volatility_lab.data.providers.base import MarketDataProvider
from volatility_lab.utils.reproducibility import seed_everything


class SyntheticMarketDataProvider(MarketDataProvider):
    """Generates synthetic OHLCV data using a discretized Heston stochastic volatility model."""

    def __init__(
        self,
        s0: float = 10000.0,
        v0: float = 0.04,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        seed: int = 42,
    ):
        self.s0 = s0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.seed = seed

    def fetch_data(
        self,
        ticker: str = "SYNTHETIC_VOL",
        start_date: Optional[str] = "2018-01-01",
        end_date: Optional[str] = "2023-12-31",
        interval: str = "1d",
    ) -> pd.DataFrame:
        seed_everything(self.seed)
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        n = len(dates)
        dt = 1.0 / 252.0

        # Correlated Brownian increments
        z1 = np.random.normal(0, 1, n)
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, 1, n)

        s = np.zeros(n)
        v = np.zeros(n)
        s[0] = self.s0
        v[0] = self.v0

        for t in range(1, n):
            # Variance process (Euler with reflection for positivity)
            v_prev = max(v[t - 1], 1e-6)
            dv = self.kappa * (self.theta - v_prev) * dt + self.xi * np.sqrt(v_prev * dt) * z2[t]
            v[t] = max(v_prev + dv, 1e-6)

            # Asset price process
            ds = np.sqrt(v_prev * dt) * z1[t]
            s[t] = s[t - 1] * np.exp(-0.5 * v_prev * dt + ds)

        # Synthesize realistic High, Low, Open, Close from daily variance
        daily_sigma = np.sqrt(v * dt)
        open_prices = s * np.exp(np.random.normal(0, daily_sigma * 0.2, n))
        high_prices = np.maximum(open_prices, s) * (1.0 + np.abs(np.random.normal(0, daily_sigma * 0.8, n)))
        low_prices = np.minimum(open_prices, s) * (1.0 - np.abs(np.random.normal(0, daily_sigma * 0.8, n)))
        volumes = np.random.lognormal(14.0, 0.5, n) * (1.0 + (v / self.theta))

        df = pd.DataFrame(
            {
                "Open": open_prices,
                "High": high_prices,
                "Low": low_prices,
                "Close": s,
                "Adj Close": s,
                "Volume": volumes,
                "True_Variance": v,
            },
            index=dates,
        )
        df.index.name = "Date"
        return df
