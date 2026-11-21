"""Return-based feature extraction for volatility modeling."""

from typing import List
import numpy as np
import pandas as pd


def extract_return_features(
    log_returns: pd.Series,
    lags: List[int] = [1, 2, 3, 5, 10, 20],
) -> pd.DataFrame:
    """Extracts strictly backward-looking return features."""
    features = {}
    r = log_returns.copy()

    for lag in lags:
        features[f"ret_lag_{lag}"] = r.shift(lag - 1)  # if lag=1, uses r_t
        features[f"abs_ret_lag_{lag}"] = np.abs(r.shift(lag - 1))
        features[f"sq_ret_lag_{lag}"] = (r.shift(lag - 1)) ** 2

    # Cumulative return momentum
    for w in [5, 10, 20, 60]:
        features[f"cum_ret_{w}d"] = r.rolling(window=w).sum()

    return pd.DataFrame(features, index=log_returns.index)
