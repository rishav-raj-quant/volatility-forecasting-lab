"""Quantile regression for predictive uncertainty and confidence bounds."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


class VolatilityQuantileForecaster:
    """Trains quantile regression estimators (default 10%, 50%, 90%) to provide predictive intervals."""

    def __init__(
        self,
        quantiles: List[float] = [0.10, 0.50, 0.90],
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        random_state: int = 42,
    ):
        self.quantiles = quantiles
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.models_: Dict[float, GradientBoostingRegressor] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "VolatilityQuantileForecaster":
        """Fits separate quantile gradient boosting models for each target alpha."""
        for q in self.quantiles:
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            model.fit(X, y)
            self.models_[q] = model
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns DataFrame of quantile predictions [q_0.10, q_0.50, q_0.90]."""
        preds = {}
        for q, model in self.models_.items():
            preds[f"q_{int(q*100):02d}"] = np.maximum(model.predict(X), 1e-6)
        return pd.DataFrame(preds, index=X.index)
