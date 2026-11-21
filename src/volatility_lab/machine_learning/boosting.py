"""Gradient Boosting, XGBoost, and LightGBM models."""

from typing import Optional
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


def get_gradient_boosting_model(
    n_estimators: int = 120,
    learning_rate: float = 0.03,
    max_depth: int = 4,
    random_state: int = 42,
) -> GradientBoostingRegressor:
    """Returns scikit-learn GradientBoostingRegressor."""
    return GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        loss="squared_error",
    )


def get_hist_gradient_boosting_model(
    max_iter: int = 120,
    learning_rate: float = 0.03,
    max_depth: Optional[int] = 4,
    random_state: int = 42,
) -> HistGradientBoostingRegressor:
    """Returns scikit-learn HistGradientBoostingRegressor."""
    return HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )


def get_xgboost_model(
    n_estimators: int = 120,
    learning_rate: float = 0.03,
    max_depth: int = 4,
    subsample: float = 0.8,
    random_state: int = 42,
):
    """Returns XGBRegressor if xgboost is installed, else falls back to GradientBoostingRegressor."""
    if XGB_AVAILABLE:
        return XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
            n_jobs=-1,
        )
    return get_gradient_boosting_model(n_estimators, learning_rate, max_depth, random_state)


def get_lightgbm_model(
    n_estimators: int = 120,
    learning_rate: float = 0.03,
    max_depth: int = 4,
    subsample: float = 0.8,
    random_state: int = 42,
):
    """Returns LGBMRegressor if lightgbm is installed, else falls back to HistGradientBoostingRegressor."""
    if LGBM_AVAILABLE:
        return LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )
    return get_hist_gradient_boosting_model(n_estimators, learning_rate, max_depth, random_state)
