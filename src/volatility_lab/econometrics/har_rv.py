"""Heterogeneous Autoregressive model of Realized Volatility (HAR-RV, Corsi 2009)."""

from typing import Optional, Dict
import numpy as np
import pandas as pd
import statsmodels.api as sm
from volatility_lab.utils.calendar import SQRT_252
from volatility_lab.utils.logging import get_logger

logger = get_logger(__name__)


class HARRVModel:
    """Corsi (2009) Heterogeneous Autoregressive Realized Volatility (HAR-RV) Model.

    RV_{t+h} = beta_0 + beta_d * RV^{(d)}_t + beta_w * RV^{(w)}_t + beta_m * RV^{(m)}_t + epsilon_{t+h}
    Mimics long-memory volatility dynamics through cascading multi-frequency components.
    Estimated via OLS with Newey-West HAC robust standard errors.
    """

    def __init__(
        self,
        daily_lag: int = 1,
        weekly_lag: int = 5,
        monthly_lag: int = 22,
        with_leverage: bool = False,
    ):
        self.daily_lag = daily_lag
        self.weekly_lag = weekly_lag
        self.monthly_lag = monthly_lag
        self.with_leverage = with_leverage
        self.fitted_model_ = None
        self.coefficients_: Dict[str, float] = {}
        self.r_squared_: float = 0.0

    def fit(
        self,
        realized_variance: pd.Series,
        negative_returns: Optional[pd.Series] = None,
        forecast_horizon: int = 1,
    ) -> "HARRVModel":
        """Fits HAR-RV model predicting RV_{t+horizon} using information up to day t."""
        rv = realized_variance.dropna()
        rv_d = rv.copy()
        rv_w = rv.rolling(window=self.weekly_lag).mean()
        rv_m = rv.rolling(window=self.monthly_lag).mean()

        features = pd.DataFrame(
            {
                "rv_daily": rv_d,
                "rv_weekly": rv_w,
                "rv_monthly": rv_m,
            },
            index=rv.index,
        )

        if self.with_leverage and negative_returns is not None:
            neg_ret = np.minimum(negative_returns.reindex(rv.index).fillna(0.0), 0.0)
            features["leverage_neg_ret"] = neg_ret

        # Target: future realized variance over next horizon days
        if forecast_horizon == 1:
            target = rv.shift(-1)
        else:
            # Average daily variance over next horizon days: (1/h) * sum_{k=1}^h RV_{t+k}
            target = rv.iloc[::-1].rolling(window=forecast_horizon).mean().iloc[::-1].shift(-1)

        data = pd.concat([features, target.rename("target")], axis=1).dropna()
        X = sm.add_constant(data[features.columns])
        y = data["target"]

        # Fit with Newey-West HAC covariance (lag = max(forecast_horizon * 2, 5))
        nw_lags = max(forecast_horizon * 2, 5)
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})

        self.fitted_model_ = model
        self.coefficients_ = dict(model.params)
        self.r_squared_ = float(model.rsquared)
        logger.info(f"Fitted HAR-RV (h={forecast_horizon}): R^2={self.r_squared_:.4f}, params={self.coefficients_}")
        return self

    def forecast(
        self,
        current_rv: pd.Series,
        negative_returns: Optional[pd.Series] = None,
    ) -> float:
        """Generates 1-step or h-step forward variance forecast using latest historical data."""
        if self.fitted_model_ is None:
            raise RuntimeError("HAR-RV model must be fitted before forecasting.")

        rv = current_rv.dropna()
        rv_d = float(rv.iloc[-1])
        rv_w = float(rv.iloc[-self.weekly_lag:].mean()) if len(rv) >= self.weekly_lag else rv_d
        rv_m = float(rv.iloc[-self.monthly_lag:].mean()) if len(rv) >= self.monthly_lag else rv_w

        X_pred = [1.0, rv_d, rv_w, rv_m]
        if self.with_leverage:
            neg_ret = float(min(negative_returns.iloc[-1], 0.0)) if negative_returns is not None else 0.0
            X_pred.append(neg_ret)

        pred_var = float(np.dot(self.fitted_model_.params.values, X_pred))
        # Ensure variance non-negativity
        pred_var = max(pred_var, 1e-6)
        return pred_var

    def forecast_volatility_annualized(
        self,
        current_rv: pd.Series,
        negative_returns: Optional[pd.Series] = None,
    ) -> float:
        """Returns annualized volatility prediction in percent."""
        var = self.forecast(current_rv, negative_returns)
        return float(np.sqrt(var) * SQRT_252)
