"""Uncertainty quantification and forecast prediction intervals."""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from volatility_lab.forecasting.horizons import variance_to_annualized_volatility


class ForecastUncertaintyEstimator:
    """Computes parametric and bootstrap empirical confidence intervals for volatility forecasts."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level

    def compute_residual_intervals(
        self,
        point_prediction: float,
        training_residuals: np.ndarray,
    ) -> Dict[str, float]:
        """Calculates empirical prediction interval from in-sample forecast errors."""
        res = training_residuals[~np.isnan(training_residuals)]
        if len(res) < 10:
            margin = point_prediction * 0.25
            lower = max(point_prediction - margin, 1e-6)
            upper = point_prediction + margin
        else:
            low_err = np.percentile(res, (self.alpha / 2.0) * 100.0)
            high_err = np.percentile(res, (1.0 - self.alpha / 2.0) * 100.0)
            lower = max(point_prediction + low_err, 1e-6)
            upper = max(point_prediction + high_err, lower * 1.01)

        return {
            "point_variance": float(point_prediction),
            "lower_variance": float(lower),
            "upper_variance": float(upper),
            "point_vol_ann_pct": float(variance_to_annualized_volatility(point_prediction) * 100.0),
            "lower_vol_ann_pct": float(variance_to_annualized_volatility(lower) * 100.0),
            "upper_vol_ann_pct": float(variance_to_annualized_volatility(upper) * 100.0),
            "confidence_level": self.confidence_level,
        }
