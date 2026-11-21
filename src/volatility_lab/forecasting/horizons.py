"""Forecasting horizon definitions and target converters."""

from typing import List, Dict, Union
import numpy as np
import pandas as pd
from volatility_lab.utils.calendar import SQRT_252, ANNUALIZATION_FACTOR_DAILY

STANDARD_HORIZONS = [1, 5, 10, 20]


def variance_to_annualized_volatility(variance: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """Converts daily variance into annualized volatility in decimal units (multiply by 100 for %).

    sigma_{ann} = sqrt(variance * 252)
    """
    return np.sqrt(np.maximum(variance, 0.0) * ANNUALIZATION_FACTOR_DAILY)


def annualized_volatility_to_variance(annualized_vol: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """Converts annualized decimal volatility into daily variance units.

    variance = (sigma_{ann})^2 / 252
    """
    return (annualized_vol ** 2) / ANNUALIZATION_FACTOR_DAILY


def build_forward_targets(
    squared_returns: pd.Series,
    horizons: List[int] = STANDARD_HORIZONS,
) -> pd.DataFrame:
    """Builds forward non-overlapping/multi-horizon variance targets.

    For horizon h:
    target_h = (1 / h) * sum_{k=1}^h r^2_{t+k}
    """
    targets = {}
    sq = squared_returns.copy()

    for h in horizons:
        if h == 1:
            t_var = sq.shift(-1)
        else:
            t_var = sq.iloc[::-1].rolling(window=h).mean().iloc[::-1].shift(-1)
        targets[f"target_var_h{h}"] = t_var
        targets[f"target_vol_ann_h{h}"] = variance_to_annualized_volatility(t_var)

    return pd.DataFrame(targets, index=squared_returns.index)
