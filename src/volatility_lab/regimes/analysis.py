"""Regime transition probabilities, durations, and stratified performance."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from volatility_lab.evaluation.metrics import compute_all_metrics


class RegimeAnalyzer:
    """Analyzes Markov transition dynamics, expected durations, and stratified performance."""

    @staticmethod
    def compute_transition_matrix(regime_series: pd.Series) -> pd.DataFrame:
        """Calculates empirical 1-step Markov transition probability matrix."""
        s = regime_series.dropna().astype(int)
        unique_regimes = sorted(s.unique())
        n_regimes = len(unique_regimes)

        counts = np.zeros((n_regimes, n_regimes))
        vals = s.values

        for t in range(len(vals) - 1):
            i = vals[t]
            j = vals[t + 1]
            counts[i, j] += 1

        row_sums = counts.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        prob_matrix = counts / row_sums

        return pd.DataFrame(
            prob_matrix,
            index=[f"Regime {r}" for r in unique_regimes],
            columns=[f"To Regime {r}" for r in unique_regimes],
        )

    @staticmethod
    def compute_expected_durations(transition_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculates expected regime persistence duration in trading days: 1 / (1 - P_ii)."""
        durations = {}
        probs = transition_matrix.values
        for i in range(len(probs)):
            p_ii = probs[i, i]
            dur = 1.0 / (1.0 - p_ii) if p_ii < 1.0 else float("inf")
            durations[f"Regime {i}"] = float(dur)
        return durations

    @staticmethod
    def evaluate_by_regime(
        y_true: pd.Series,
        y_pred: pd.Series,
        regime_series: pd.Series,
    ) -> pd.DataFrame:
        """Evaluates out-of-sample forecast accuracy stratified across market regimes."""
        combined = pd.DataFrame({
            "true": y_true,
            "pred": y_pred,
            "regime": regime_series,
        }).dropna()

        records = []
        for reg, group in combined.groupby("regime"):
            m = compute_all_metrics(group["true"].values, group["pred"].values)
            m["regime"] = int(reg)
            m["observations"] = len(group)
            records.append(m)

        return pd.DataFrame(records).set_index("regime")
