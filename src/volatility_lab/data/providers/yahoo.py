"""Yahoo Finance data provider implementation."""

from typing import Optional
import pandas as pd
import yfinance as yf
from volatility_lab.data.providers.base import MarketDataProvider
from volatility_lab.utils.logging import get_logger

logger = get_logger(__name__)


class YahooFinanceProvider(MarketDataProvider):
    """Retrieves market data using yfinance API."""

    def __init__(self, auto_adjust: bool = False):
        self.auto_adjust = auto_adjust

    def fetch_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        logger.info(f"Fetching {ticker} data from Yahoo Finance [{start_date} -> {end_date}, interval={interval}]")
        try:
            df = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=self.auto_adjust,
                progress=False,
            )
            if df.empty:
                raise ValueError(f"No market data returned for ticker '{ticker}' from Yahoo Finance.")

            # Flatten MultiIndex columns if present (common in modern yfinance)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Standardize index and required columns
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"

            expected_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in expected_cols:
                if col not in df.columns:
                    raise KeyError(f"Expected column '{col}' missing from downloaded dataset.")

            if "Adj Close" not in df.columns:
                df["Adj Close"] = df["Close"]

            df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].astype(float)
            logger.info(f"Successfully retrieved {len(df)} records for {ticker}.")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            raise
