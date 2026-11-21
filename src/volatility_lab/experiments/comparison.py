"""Model leaderboard, benchmark comparisons, and statistical ranking."""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from volatility_lab.evaluation.statistical_tests import diebold_mariano_test


class ModelComparisonFramework:
    """Builds empirical model leaderboards and statistical pairwise comparisons."""

    @staticmethod
    def generate_leaderboard(
        predictions_dict: Dict[str, np.ndarray],
        y_true: np.ndarray,
        horizon: int = 1,
    ) -> pd.DataFrame:
        """Evaluates and ranks all models by QLIKE loss on the identical out-of-sample test set."""
        from volatility_lab.evaluation.metrics import compute_all_metrics

        records = []
        # Find baseline (Historical or GARCH or first model)
        benchmark_key = next((k for k in predictions_dict.keys() if "historical" in k.lower() or "garch" in k.lower()), list(predictions_dict.keys())[0])
        benchmark_preds = predictions_dict[benchmark_key]

        for name, preds in predictions_dict.items():
            m = compute_all_metrics(y_true, preds)
            dm = diebold_mariano_test(y_true, preds, benchmark_preds, horizon=horizon, loss_type="qlike")

            records.append({
                "model": name,
                "qlike": m["qlike"],
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
                "mape_pct": m["mape_pct"],
                "dm_stat_vs_bench": dm.statistic,
                "dm_pvalue": dm.p_value,
                "dm_significance": "Significant (p<0.05)" if dm.p_value < 0.05 else "Not Significant",
            })

        df = pd.DataFrame(records).sort_values("qlike", ascending=True).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df[["rank", "model", "qlike", "rmse", "mae", "r2", "dm_stat_vs_bench", "dm_pvalue", "dm_significance"]]
