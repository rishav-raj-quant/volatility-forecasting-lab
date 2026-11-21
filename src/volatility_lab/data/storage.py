"""Parquet storage manager for analytical market data and model artifacts."""

import os
from typing import Optional
import pandas as pd
from volatility_lab.utils.logging import get_logger

logger = get_logger(__name__)


class ParquetStorageManager:
    """Manages saving and loading of time-series datasets using Apache Parquet."""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "raw")
        self.processed_dir = os.path.join(base_dir, "processed")
        self.features_dir = os.path.join(base_dir, "features")
        self.predictions_dir = os.path.join(base_dir, "predictions")

        for d in [self.raw_dir, self.processed_dir, self.features_dir, self.predictions_dir]:
            os.makedirs(d, exist_ok=True)

    def _clean_name(self, name: str) -> str:
        return name.replace("^", "").replace("-", "_").replace(" ", "_")

    def save_processed(self, df: pd.DataFrame, ticker: str, interval: str = "1d") -> str:
        path = os.path.join(self.processed_dir, f"{self._clean_name(ticker)}_{interval}.parquet")
        df.to_parquet(path, compression="snappy", index=True)
        logger.info(f"Saved processed dataset ({len(df)} rows) to {path}")
        return path

    def load_processed(self, ticker: str, interval: str = "1d") -> Optional[pd.DataFrame]:
        path = os.path.join(self.processed_dir, f"{self._clean_name(ticker)}_{interval}.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None

    def save_features(self, df: pd.DataFrame, ticker: str, horizon: int = 1) -> str:
        path = os.path.join(self.features_dir, f"{self._clean_name(ticker)}_h{horizon}_features.parquet")
        df.to_parquet(path, compression="snappy", index=True)
        return path

    def load_features(self, ticker: str, horizon: int = 1) -> Optional[pd.DataFrame]:
        path = os.path.join(self.features_dir, f"{self._clean_name(ticker)}_h{horizon}_features.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None

    def save_predictions(self, df: pd.DataFrame, experiment_id: str) -> str:
        path = os.path.join(self.predictions_dir, f"{experiment_id}_predictions.parquet")
        df.to_parquet(path, compression="snappy", index=True)
        return path

    def load_predictions(self, experiment_id: str) -> Optional[pd.DataFrame]:
        path = os.path.join(self.predictions_dir, f"{experiment_id}_predictions.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None
