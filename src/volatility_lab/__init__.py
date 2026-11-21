"""
Volatility Forecasting Lab
==========================
An institutional-grade quantitative research and machine learning platform for
forecasting financial market volatility using econometric models, statistical learning,
realized volatility estimation, walk-forward validation, and quantitative risk analytics.
"""

__version__ = "1.0.0"
__author__ = "Rishav Raj"

from volatility_lab.estimators.historical import HistoricalVolatilityEstimator
from volatility_lab.estimators.ewma import EWMAVolatilityEstimator
from volatility_lab.estimators.parkinson import ParkinsonEstimator
from volatility_lab.estimators.garman_klass import GarmanKlassEstimator
from volatility_lab.estimators.rogers_satchell import RogersSatchellEstimator
from volatility_lab.estimators.realized import RealizedVolatilityEstimator
from volatility_lab.evaluation.metrics import qlike_loss, rmse, mae, mse
from volatility_lab.risk.var import ValueAtRisk
from volatility_lab.risk.expected_shortfall import ExpectedShortfall

__all__ = [
    "HistoricalVolatilityEstimator",
    "EWMAVolatilityEstimator",
    "ParkinsonEstimator",
    "GarmanKlassEstimator",
    "RogersSatchellEstimator",
    "RealizedVolatilityEstimator",
    "qlike_loss",
    "rmse",
    "mae",
    "mse",
    "ValueAtRisk",
    "ExpectedShortfall",
]
