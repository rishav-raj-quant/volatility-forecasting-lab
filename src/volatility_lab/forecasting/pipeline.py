"""Unified volatility forecasting pipeline."""

from typing import Dict, Any, List, Optional
import pandas as pd
from volatility_lab.forecasting.prediction import MultiModelForecaster
from volatility_lab.forecasting.horizons import STANDARD_HORIZONS
from volatility_lab.utils.logging import get_logger

logger = get_logger(__name__)


class VolatilityForecastingPipeline:
    """End-to-end pipeline generating and aggregating multi-horizon forecasts."""

    def __init__(self, horizons: List[int] = STANDARD_HORIZONS):
        self.horizons = horizons
        self.forecaster = MultiModelForecaster(horizons=horizons)

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes full multi-horizon multi-model forecasting procedure."""
        logger.info(f"Running multi-model forecasting pipeline for horizons {self.horizons}...")
        results = self.forecaster.forecast_all(df)
        logger.info(f"Generated {len(results)} volatility forecast points.")
        return results
