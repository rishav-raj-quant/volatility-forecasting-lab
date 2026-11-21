"""Random Forest and Extra Trees regression models."""

from typing import Optional
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_random_forest_model(
    n_estimators: int = 150,
    max_depth: Optional[int] = 6,
    min_samples_split: int = 5,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Returns configured RandomForestRegressor for volatility prediction."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )


def get_extra_trees_model(
    n_estimators: int = 150,
    max_depth: Optional[int] = 6,
    min_samples_split: int = 5,
    random_state: int = 42,
) -> ExtraTreesRegressor:
    """Returns configured ExtraTreesRegressor for volatility prediction."""
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )
