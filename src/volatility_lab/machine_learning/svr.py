"""Support Vector Regression (SVR) for volatility forecasting."""

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_svr_model(
    kernel: str = "rbf",
    C: float = 1.0,
    epsilon: float = 0.05,
    gamma: str = "scale",
) -> Pipeline:
    """Returns Pipeline with StandardScaler and SVR."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma))
    ])
