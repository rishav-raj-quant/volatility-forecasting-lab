"""Leakage-free feature pipeline construction."""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from volatility_lab.features.returns import extract_return_features
from volatility_lab.features.volatility import extract_volatility_features
from volatility_lab.features.technical import extract_technical_features
from volatility_lab.utils.calendar import SQRT_252, ANNUALIZATION_FACTOR_DAILY
from volatility_lab.utils.logging import get_logger

logger = get_logger(__name__)


class FeaturePipeline:
    """Orchestrates comprehensive feature engineering with strict leakage prevention."""

    def __init__(self, target_horizons: List[int] = [1, 5, 10, 20]):
        self.target_horizons = target_horizons

    def transform(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Transforms clean market data into feature matrix X and target series y.

        Target y_t: Forward average daily realized variance over [t+1, t+horizon]:
            y_t = (1 / horizon) * sum_{k=1}^{horizon} r^2_{t+k}

        Strictly guarantees that row t contains features computed using data up to t,
        while the target represents unseen future variance from t+1 onwards.
        """
        if "log_return" not in df.columns:
            raise KeyError("DataFrame must contain 'log_return' column.")

        f_ret = extract_return_features(df["log_return"])
        f_vol = extract_volatility_features(df)
        f_tech = extract_technical_features(df)

        X = pd.concat([f_ret, f_vol, f_tech], axis=1)

        # Forward target variance calculation
        # r_{t+1}^2 to r_{t+horizon}^2
        sq_ret = df["log_return"] ** 2

        if horizon == 1:
            y = sq_ret.shift(-1)
        else:
            # Average daily variance over horizon: shift(-horizon) rolling mean inverted
            # Forward rolling mean of length horizon:
            y = sq_ret.iloc[::-1].rolling(window=horizon).mean().iloc[::-1].shift(-1)

        # Name target
        y.name = f"target_var_h{horizon}"

        # Align and drop rows with NaN (warmup period and last horizon rows)
        combined = pd.concat([X, y], axis=1).dropna()
        X_clean = combined[X.columns]
        y_clean = combined[y.name]

        logger.info(f"Built feature matrix for horizon={horizon}: {X_clean.shape[0]} samples, {X_clean.shape[1]} features.")
        return X_clean, y_clean

    def transform_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms latest available historical data for out-of-sample forward prediction.
        Returns single-row or tail feature matrix where target is unknown.
        """
        f_ret = extract_return_features(df["log_return"])
        f_vol = extract_volatility_features(df)
        f_tech = extract_technical_features(df)

        X = pd.concat([f_ret, f_vol, f_tech], axis=1)
        return X.dropna()
