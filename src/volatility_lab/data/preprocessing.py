"""Market data preprocessing and return generation module."""

import numpy as np
import pandas as pd
from typing import Optional
from volatility_lab.utils.logging import get_logger
from volatility_lab.utils.calendar import ensure_market_timezone

logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocesses raw financial price series into clean research-grade analytical series."""

    def __init__(
        self,
        target_tz: str = "Asia/Kolkata",
        max_ffill_limit: int = 3,
        drop_na_returns: bool = True,
    ):
        self.target_tz = target_tz
        self.max_ffill_limit = max_ffill_limit
        self.drop_na_returns = drop_na_returns

    def clean_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans OHLCV series, removes duplicates, handles missing values, and adds core returns."""
        clean = df.copy()

        # Remove duplicate timestamps keeping the latest observation
        clean = clean[~clean.index.duplicated(keep="last")]
        clean = clean.sort_index()

        # Ensure market timezone if intraday or localized
        if isinstance(clean.index, pd.DatetimeIndex) and clean.index.tz is not None:
            clean.index = ensure_market_timezone(clean.index, self.target_tz)

        # Forward fill up to limit, drop remaining missing rows
        clean = clean.ffill(limit=self.max_ffill_limit)
        clean = clean.dropna(subset=["Close"])

        # Price sanity clipping (strictly positive)
        for col in ["Open", "High", "Low", "Close"]:
            if col in clean.columns:
                clean[col] = np.maximum(clean[col], 1e-4)

        # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
        if {"High", "Low", "Open", "Close"}.issubset(clean.columns):
            clean["High"] = np.maximum(clean["High"], np.maximum(clean["Open"], clean["Close"]))
            clean["Low"] = np.minimum(clean["Low"], np.minimum(clean["Open"], clean["Close"]))

        # Core Log Returns
        # r_t = ln(P_t / P_{t-1})
        clean["log_return"] = np.log(clean["Close"] / clean["Close"].shift(1))
        clean["simple_return"] = clean["Close"].pct_change()

        # Overnight and Intraday returns
        if "Open" in clean.columns:
            clean["overnight_return"] = np.log(clean["Open"] / clean["Close"].shift(1))
            clean["intraday_return"] = np.log(clean["Close"] / clean["Open"])

        # Daily Variance Proxy: r_t^2 (unbiased daily squared return proxy)
        clean["daily_variance_proxy"] = clean["log_return"] ** 2

        if self.drop_na_returns:
            clean = clean.dropna(subset=["log_return"])

        return clean
