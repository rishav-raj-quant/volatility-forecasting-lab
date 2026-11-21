"""Technical, trend, drawdown, and volume indicator features."""

import numpy as np
import pandas as pd


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Computes Wilders Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()

    rs = gain / np.maximum(loss, 1e-6)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Computes Average True Range normalized by close price."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr / close


def extract_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts technical momentum, trend, drawdown, and volume features."""
    features = pd.DataFrame(index=df.index)
    close = df["Close"]

    # Moving Average Distance
    for w in [20, 50, 200]:
        if len(close) >= w:
            sma = close.rolling(window=w).mean()
            features[f"dist_sma_{w}"] = (close - sma) / sma

    # RSI
    features["rsi_14"] = compute_rsi(close, period=14)

    # ATR
    if {"High", "Low", "Close"}.issubset(df.columns):
        features["atr_norm_14"] = compute_atr(df["High"], df["Low"], df["Close"], period=14)

    # Rolling Maximum Drawdown
    for w in [20, 60]:
        roll_max = close.rolling(window=w, min_periods=w).max()
        features[f"drawdown_{w}d"] = (close - roll_max) / roll_max

    # Volume dynamics if volume present
    if "Volume" in df.columns:
        vol = df["Volume"].replace(0, np.nan).ffill()
        features["log_volume"] = np.log(vol)
        vol_mean = vol.rolling(window=20).mean()
        vol_std = vol.rolling(window=20).std().replace(0, 1.0)
        features["volume_zscore_20d"] = (vol - vol_mean) / vol_std

    return features
