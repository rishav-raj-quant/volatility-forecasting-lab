"""Multi-model volatility prediction orchestrator across forecasting horizons."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from volatility_lab.forecasting.horizons import variance_to_annualized_volatility, annualized_volatility_to_variance
from volatility_lab.forecasting.uncertainty import ForecastUncertaintyEstimator
from volatility_lab.estimators.historical import HistoricalVolatilityEstimator
from volatility_lab.estimators.ewma import EWMAVolatilityEstimator
from volatility_lab.econometrics.garch import GARCHModel
from volatility_lab.econometrics.har_rv import HARRVModel
from volatility_lab.features.feature_pipeline import FeaturePipeline
from volatility_lab.machine_learning.registry import MLModelRegistry
from volatility_lab.utils.logging import get_logger

logger = get_logger(__name__)


class MultiModelForecaster:
    """Trains and generates volatility forecasts across multiple model classes and horizons."""

    def __init__(self, horizons: List[int] = [1, 5, 10, 20]):
        self.horizons = horizons
        self.uncertainty = ForecastUncertaintyEstimator(confidence_level=0.95)

    def forecast_all(
        self,
        df: pd.DataFrame,
        ml_models: List[str] = ["ridge", "random_forest", "lightgbm"],
    ) -> pd.DataFrame:
        """Generates comprehensive multi-horizon forecast matrix comparing econometric & ML models."""
        returns = df["log_return"].dropna()
        records = []

        # 1. Historical Volatility Benchmark (20-day rolling)
        hist_20d_ann = HistoricalVolatilityEstimator(windows=20, annualize=True).estimate(returns).iloc[-1, 0]
        hist_20d_var = annualized_volatility_to_variance(hist_20d_ann)

        for h in self.horizons:
            records.append({
                "model": "Historical (20d)",
                "category": "Benchmark",
                "horizon_days": h,
                "forecast_variance": float(hist_20d_var),
                "forecast_vol_ann_pct": float(hist_20d_ann * 100.0),
                "lower_vol_ann_pct": float(hist_20d_ann * 0.85 * 100.0),
                "upper_vol_ann_pct": float(hist_20d_ann * 1.15 * 100.0),
            })

        # 2. EWMA (RiskMetrics lambda=0.94)
        ewma_ann = EWMAVolatilityEstimator(decay_factor=0.94, annualize=True).estimate(returns).iloc[-1]
        ewma_var = annualized_volatility_to_variance(ewma_ann)
        for h in self.horizons:
            records.append({
                "model": "EWMA (lambda=0.94)",
                "category": "Benchmark",
                "horizon_days": h,
                "forecast_variance": float(ewma_var),
                "forecast_vol_ann_pct": float(ewma_ann * 100.0),
                "lower_vol_ann_pct": float(ewma_ann * 0.85 * 100.0),
                "upper_vol_ann_pct": float(ewma_ann * 1.15 * 100.0),
            })

        # 3. GARCH(1,1)
        try:
            garch = GARCHModel(p=1, q=1).fit(returns)
            for h in self.horizons:
                vol_ann = garch.forecast(horizon=h)
                var_h = annualized_volatility_to_variance(vol_ann)
                records.append({
                    "model": "GARCH(1,1)",
                    "category": "Econometric",
                    "horizon_days": h,
                    "forecast_variance": float(var_h),
                    "forecast_vol_ann_pct": float(vol_ann * 100.0),
                    "lower_vol_ann_pct": float(vol_ann * 0.88 * 100.0),
                    "upper_vol_ann_pct": float(vol_ann * 1.12 * 100.0),
                })
        except Exception as e:
            logger.warning(f"GARCH fitting failed: {e}")

        # 4. HAR-RV
        try:
            rv_proxy = df["daily_variance_proxy"]
            for h in self.horizons:
                har = HARRVModel().fit(rv_proxy, forecast_horizon=h)
                var_fc = har.forecast(rv_proxy)
                vol_ann = variance_to_annualized_volatility(var_fc)
                records.append({
                    "model": "HAR-RV (Corsi)",
                    "category": "Econometric",
                    "horizon_days": h,
                    "forecast_variance": float(var_fc),
                    "forecast_vol_ann_pct": float(vol_ann * 100.0),
                    "lower_vol_ann_pct": float(vol_ann * 0.88 * 100.0),
                    "upper_vol_ann_pct": float(vol_ann * 1.12 * 100.0),
                })
        except Exception as e:
            logger.warning(f"HAR-RV fitting failed: {e}")

        # 5. Machine Learning Models
        pipe = FeaturePipeline()
        for h in self.horizons:
            try:
                X, y = pipe.transform(df, horizon=h)
                if len(X) < 50:
                    continue

                # Latest row for inference
                latest_X = pipe.transform_inference(df).iloc[[-1]]
                common_cols = [c for c in X.columns if c in latest_X.columns]
                X_train = X[common_cols]
                latest_X = latest_X[common_cols]

                for ml_name in ml_models:
                    model = MLModelRegistry.get(ml_name)
                    model.fit(X_train, y)
                    pred_var = float(max(model.predict(latest_X)[0], 1e-6))
                    pred_vol_ann = float(variance_to_annualized_volatility(pred_var))

                    # In-sample residuals for uncertainty bounds
                    in_sample_preds = model.predict(X_train)
                    residuals = y.values - in_sample_preds
                    unc = self.uncertainty.compute_residual_intervals(pred_var, residuals)

                    records.append({
                        "model": ml_name.upper().replace("_", " "),
                        "category": "Machine Learning",
                        "horizon_days": h,
                        "forecast_variance": pred_var,
                        "forecast_vol_ann_pct": pred_vol_ann * 100.0,
                        "lower_vol_ann_pct": unc["lower_vol_ann_pct"],
                        "upper_vol_ann_pct": unc["upper_vol_ann_pct"],
                    })
            except Exception as e:
                logger.warning(f"ML model {ml_name} (h={h}) failed: {e}")

        res_df = pd.DataFrame(records)
        return res_df
