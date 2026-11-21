"""Local CSV and Parquet file market data provider."""

import os
from typing import Optional
import pandas as pd
from volatility_lab.data.providers.base import MarketDataProvider
from volatility_lab.utils.logging import get_logger

logger = get_logger(__name__)


class LocalFileProvider(MarketDataProvider):
    """Loads market data from local files (CSV or Parquet)."""

    def __init__(self, base_dir: str = "data/raw"):
        self.base_dir = base_dir

    def fetch_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        clean_ticker = ticker.replace("^", "").replace("-", "_")
        csv_path = os.path.join(self.base_dir, f"{clean_ticker}_{interval}.csv")
        parquet_path = os.path.join(self.base_dir, f"{clean_ticker}_{interval}.parquet")

        if os.path.exists(parquet_path):
            logger.info(f"Loading data from Parquet file: {parquet_path}")
            df = pd.read_parquet(parquet_path)
        elif os.path.exists(csv_path):
            logger.info(f"Loading data from CSV file: {csv_path}")
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        else:
            raise FileNotFoundError(f"No local data found for ticker '{ticker}' in '{self.base_dir}'.")

        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        return df
