"""Script to acquire and cache market data."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from volatility_lab.data.ingestion import DataIngestionPipeline
from volatility_lab.utils.logging import get_logger

logger = get_logger("download_script")


def main():
    tickers = ["^NSEI", "^NSEBANK", "^GSPC", "BTC-USD"]
    pipeline = DataIngestionPipeline()

    for ticker in tickers:
        logger.info(f"Ingesting market data for {ticker}...")
        try:
            df = pipeline.run(ticker=ticker, start_date="2015-01-01", end_date="2024-12-31", use_cache=False)
            logger.info(f"Successfully processed {ticker}: {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
        except Exception as e:
            logger.error(f"Failed to ingest {ticker}: {e}")


if __name__ == "__main__":
    main()
