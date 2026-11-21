"""Leakage-safe walk-forward time-series validation with purging and embargo."""

from typing import List, Dict, Any, Generator, Tuple
import numpy as np
import pandas as pd
from volatility_lab.evaluation.metrics import compute_all_metrics
from volatility_lab.utils.logging import get_logger

logger = get_logger(__name__)


class WalkForwardValidator:
    """Implements strictly time-ordered walk-forward cross-validation for time series."""

    def __init__(
        self,
        scheme: str = "expanding",
        min_train_size: int = 500,
        test_size: int = 126,
        step_size: int = 126,
        embargo_size: int = 0,
    ):
        self.scheme = scheme.lower()
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.embargo_size = embargo_size

    def split(
        self,
        n_samples: int,
        forecast_horizon: int = 1,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generates (train_indices, test_indices) tuples.
        Purges overlapping samples between train and test based on forecast_horizon and embargo_size.
        """
        purge_len = max(forecast_horizon - 1, 0) + self.embargo_size

        start_train = 0
        current_train_end = self.min_train_size

        while current_train_end + purge_len + self.test_size <= n_samples:
            if self.scheme == "expanding":
                train_idx = np.arange(0, current_train_end)
            elif self.scheme == "rolling":
                train_idx = np.arange(current_train_end - self.min_train_size, current_train_end)
            else:
                raise ValueError(f"Unknown walk-forward scheme: {self.scheme}")

            test_start = current_train_end + purge_len
            test_end = test_start + self.test_size
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx
            current_train_end += self.step_size

    def evaluate_model(
        self,
        model_factory,
        X: pd.DataFrame,
        y: pd.Series,
        forecast_horizon: int = 1,
    ) -> Dict[str, Any]:
        """Runs walk-forward backtest and aggregates out-of-sample metrics."""
        y_true_all = []
        y_pred_all = []
        fold_results = []

        fold = 1
        for train_idx, test_idx in self.split(len(X), forecast_horizon=forecast_horizon):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

            # Fit model on training slice ONLY
            model = model_factory()
            model.fit(X_tr, y_tr)
            preds = np.maximum(model.predict(X_te), 1e-6)

            metrics_fold = compute_all_metrics(y_te.values, preds)
            metrics_fold["fold"] = fold
            metrics_fold["test_start"] = str(y_te.index[0].date())
            metrics_fold["test_end"] = str(y_te.index[-1].date())
            fold_results.append(metrics_fold)

            y_true_all.extend(y_te.values)
            y_pred_all.extend(preds)
            fold += 1

        overall_metrics = compute_all_metrics(np.array(y_true_all), np.array(y_pred_all))
        logger.info(f"Walk-Forward ({fold - 1} folds) completed. Overall QLIKE: {overall_metrics['qlike']:.4f}, RMSE: {overall_metrics['rmse']:.6f}")

        return {
            "overall_metrics": overall_metrics,
            "fold_metrics": pd.DataFrame(fold_results),
            "y_true": np.array(y_true_all),
            "y_pred": np.array(y_pred_all),
            "total_folds": fold - 1,
        }
