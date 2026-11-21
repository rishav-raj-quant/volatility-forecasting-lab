"""Linear regularized models for volatility forecasting."""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def get_linear_model(
    model_type: str = "ridge",
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    random_state: int = 42,
) -> Pipeline:
    """Returns scikit-learn Pipeline with StandardScaler and regularized linear model.
    StandardScaler fits strictly on training data during pipeline fit.
    """
    m_type = model_type.lower()
    if m_type == "ridge":
        reg = Ridge(alpha=alpha, random_state=random_state)
    elif m_type == "lasso":
        reg = Lasso(alpha=alpha, random_state=random_state, max_iter=2000)
    elif m_type in ("elastic_net", "elasticnet"):
        reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=2000)
    else:
        raise ValueError(f"Unsupported linear model type: {model_type}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", reg)
    ])
