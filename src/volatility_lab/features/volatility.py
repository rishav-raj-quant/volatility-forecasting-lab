"""Rolling volatility, vol-of-vol, and distribution moment features."""

from typing import List
import numpy as np
import pandas as pd
from volatility_lab.estimators.historical import HistoricalVolatilityEstimator
from volatility_lab.estimators.ewma import EWMAVolatilityEstimator
from volatility_lab.estimators.parkinson import ParkinsonEstimator
from volatility_lab.estimators.garman_klass import GarmanKlassEstimator
from volatility_lab.estimators.rogers_satchell import RogersSatchellEstimator


def extract_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts range-based, historical, EWMA, and vol-of-vol features."""
    features = pd.DataFrame(index=df.index)
    returns = df["log_return"]

    # Historical Close-to-Close volatility
    hist = HistoricalVolatilityEstimator(windows=[5, 10, 20, 60], annualize=False).estimate(returns)
    features = pd.concat([features, hist], axis=1)

    # EWMA volatility
    features["ewma_vol_94"] = EWMAVolatilityEstimator(decay_factor=0.94, annualize=False).estimate(returns)
    features["ewma_vol_97"] = EWMAVolatilityEstimator(decay_factor=0.97, annualize=False).estimate(returns)

    # Range-based estimators if OHLC present
    if {"Open", "High", "Low", "Close"}.issubset(df.columns):
        p_est = ParkinsonEstimator(window=20, annualize=False).estimate(df["High"], df["Low"])
        gk_est = GarmanKlassEstimator(window=20, annualize=False).estimate(df["Open"], df["High"], df["Low"], df["Close"])
        rs_est = RogersSatchellEstimator(window=20, annualize=False).estimate(df["Open"], df["High"], df["Low"], df["Close"])
        features["parkinson_20d"] = p_est
        features["garman_klass_20d"] = gk_est
        features["rogers_satchell_20d"] = rs_est

        # Daily Range
        features["high_low_ratio"] = np.log(df["High"] / df["Low"])
        features["close_open_ratio"] = np.log(df["Close"] / df["Open"])

    # Volatility of Volatility (rolling standard deviation of 20d volatility)
    if "hist_vol_20d" in features.columns:
        features["vol_of_vol_20d"] = features["hist_vol_20d"].rolling(window=20).std()

    # Rolling Skewness and Kurtosis
    features["skew_20d"] = returns.rolling(window=20).skew()
    features["kurt_20d"] = returns.rolling(window=20).kurt()

    return features
