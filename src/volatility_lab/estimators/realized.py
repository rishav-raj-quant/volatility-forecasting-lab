"""Realized volatility, bipower variation, and multi-scale aggregation module."""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from volatility_lab.utils.calendar import SQRT_252


class RealizedVolatilityEstimator:
    """Computes realized variance, realized volatility, and jump diagnostics."""

    def __init__(self, annualize: bool = True):
        self.annualize = annualize

    def from_intraday_returns(self, intraday_returns: pd.Series) -> pd.DataFrame:
        """Computes daily realized variance, bipower variation, and jump components from intraday returns.

        Args:
            intraday_returns: Intraday log returns with DateTimeIndex.
        """
        # Group by trading date
        daily_groups = intraday_returns.groupby(intraday_returns.index.date)

        records = []
        c_const = np.pi / 2.0

        for date_val, group in daily_groups:
            r = group.dropna().values
            m = len(r)
            if m < 2:
                continue

            rv = np.sum(r ** 2)
            # Bipower variation (Barndorff-Nielsen & Shephard, 2004)
            bv = c_const * (m / (m - 1)) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
            jump = max(rv - bv, 0.0)

            records.append({
                "Date": pd.to_datetime(date_val),
                "realized_variance": rv,
                "bipower_variation": bv,
                "jump_component": jump,
                "observation_count": m,
            })

        df = pd.DataFrame(records).set_index("Date")
        factor = SQRT_252 if self.annualize else 1.0
        df["realized_volatility"] = np.sqrt(df["realized_variance"]) * factor
        return df

    def compute_har_components(
        self,
        daily_rv: pd.Series,
        daily_lag: int = 1,
        weekly_lag: int = 5,
        monthly_lag: int = 22,
    ) -> pd.DataFrame:
        """Aggregates daily realized variance into HAR components: daily, weekly, monthly averages.

        RV^{(d)}_t = RV_t
        RV^{(w)}_t = 1/5 * sum_{i=0}^4 RV_{t-i}
        RV^{(m)}_t = 1/22 * sum_{i=0}^{21} RV_{t-i}
        """
        rv_d = daily_rv.shift(daily_lag)
        rv_w = daily_rv.rolling(window=weekly_lag).mean().shift(daily_lag)
        rv_m = daily_rv.rolling(window=monthly_lag).mean().shift(daily_lag)

        return pd.DataFrame(
            {
                "rv_daily": rv_d,
                "rv_weekly": rv_w,
                "rv_monthly": rv_m,
            },
            index=daily_rv.index,
        )
