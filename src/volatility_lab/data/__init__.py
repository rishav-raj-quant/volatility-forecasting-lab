"""Data acquisition, validation, preprocessing, storage, and real-time streaming package."""

from volatility_lab.data.providers.base import MarketDataProvider
from volatility_lab.data.providers.yahoo import YahooFinanceProvider
from volatility_lab.data.providers.csv_provider import LocalFileProvider
from volatility_lab.data.providers.synthetic import SyntheticMarketDataProvider
from volatility_lab.data.validation import DataValidator, ValidationReport
from volatility_lab.data.preprocessing import DataPreprocessor
from volatility_lab.data.storage import ParquetStorageManager
from volatility_lab.data.ingestion import DataIngestionPipeline
from volatility_lab.data.realtime import (
    RealTimeVolatilityEngine,
    LiveStreamingSimulator,
    compute_realtime_forecast_cones,
)

__all__ = [
    "MarketDataProvider",
    "YahooFinanceProvider",
    "LocalFileProvider",
    "SyntheticMarketDataProvider",
    "DataValidator",
    "ValidationReport",
    "DataPreprocessor",
    "ParquetStorageManager",
    "DataIngestionPipeline",
    "RealTimeVolatilityEngine",
    "LiveStreamingSimulator",
    "compute_realtime_forecast_cones",
]
