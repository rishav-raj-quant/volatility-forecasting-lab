"""Abstract base class for financial market data providers."""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class MarketDataProvider(ABC):
    """Abstract interface defining market data retrieval contract."""

    @abstractmethod
    def fetch_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Retrieves raw OHLCV market data.

        Returns DataFrame indexed by DateTimeIndex with standardized columns:
        ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'].
        """
        pass
