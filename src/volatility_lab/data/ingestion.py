"""Integrated market data ingestion orchestrator."""

from typing import Optional
import pandas as pd
from volatility_lab.data.providers.base import MarketDataProvider
from volatility_lab.data.providers.yahoo import YahooFinanceProvider
from volatility_lab.data.providers.synthetic import SyntheticMarketDataProvider
from volatility_lab.data.validation import DataValidator, ValidationReport
from volatility_lab.data.preprocessing import DataPreprocessor
from volatility_lab.data.storage import ParquetStorageManager
from volatility_lab.utils.logging import get_logger

logger = get_logger(__name__)


class DataIngestionPipeline:
    """End-to-end data ingestion, validation, and storage pipeline."""

    def __init__(
        self,
        provider: Optional[MarketDataProvider] = None,
        validator: Optional[DataValidator] = None,
        preprocessor: Optional[DataPreprocessor] = None,
        storage: Optional[ParquetStorageManager] = None,
    ):
        self.provider = provider or YahooFinanceProvider()
        self.validator = validator or DataValidator()
        self.preprocessor = preprocessor or DataPreprocessor()
        self.storage = storage or ParquetStorageManager()

    def run(
        self,
        ticker: str = "^NSEI",
        start_date: Optional[str] = "2015-01-01",
        end_date: Optional[str] = "2024-12-31",
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Executes full pipeline: cache check -> fetch -> validate -> preprocess -> cache -> return."""
        if use_cache:
            cached = self.storage.load_processed(ticker, interval)
            if cached is not None and len(cached) > 0:
                logger.info(f"Loaded {len(cached)} cached processed rows for {ticker}.")
                return cached

        try:
            raw_df = self.provider.fetch_data(ticker, start_date, end_date, interval)
        except Exception as e:
            logger.warning(f"Failed to fetch data using primary provider: {e}. Falling back to synthetic simulation.")
            synth = SyntheticMarketDataProvider()
            raw_df = synth.fetch_data(ticker, start_date, end_date, interval)

        report = self.validator.validate(raw_df)
        if not report.is_valid:
            logger.warning(f"Data validation reported issues for {ticker}: {report.issues}")

        processed_df = self.preprocessor.clean_and_prepare(raw_df)
        self.storage.save_processed(processed_df, ticker, interval)
        return processed_df
