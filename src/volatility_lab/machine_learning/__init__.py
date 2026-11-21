"""Machine learning models package for financial volatility forecasting."""

from volatility_lab.machine_learning.registry import MLModelRegistry
from volatility_lab.machine_learning.linear import get_linear_model
from volatility_lab.machine_learning.tree_models import get_random_forest_model, get_extra_trees_model
from volatility_lab.machine_learning.boosting import (
    get_gradient_boosting_model,
    get_hist_gradient_boosting_model,
    get_xgboost_model,
    get_lightgbm_model,
)
from volatility_lab.machine_learning.svr import get_svr_model
from volatility_lab.machine_learning.quantile import VolatilityQuantileForecaster

__all__ = [
    "MLModelRegistry",
    "get_linear_model",
    "get_random_forest_model",
    "get_extra_trees_model",
    "get_gradient_boosting_model",
    "get_hist_gradient_boosting_model",
    "get_xgboost_model",
    "get_lightgbm_model",
    "get_svr_model",
    "VolatilityQuantileForecaster",
]
