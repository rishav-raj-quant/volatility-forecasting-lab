"""Unit tests for volatility regime detection and Markov transition analysis."""

import pytest
import numpy as np
import pandas as pd
from volatility_lab.regimes.detection import StatisticalRegimeDetector, GMMRegimeDetector
from volatility_lab.regimes.analysis import RegimeAnalyzer


@pytest.fixture
def variance_series():
    dates = pd.date_range("2021-01-01", periods=600, freq="B")
    np.random.seed(42)
    # Simulate regime switching variance
    var_low = np.random.exponential(0.01, 200)
    var_high = np.random.exponential(0.08, 200)
    var_extreme = np.random.exponential(0.20, 200)
    var = np.concatenate([var_low, var_high, var_extreme])
    return pd.Series(var, index=dates)


def test_statistical_regime_detector(variance_series):
    detector = StatisticalRegimeDetector(warmup_period=200)
    regimes = detector.detect(variance_series)

    assert len(regimes) == len(variance_series)
    unique_states = sorted(regimes.unique())
    assert set(unique_states).issubset({0, 1, 2, 3})


def test_gmm_regime_detector(variance_series):
    gmm_det = GMMRegimeDetector(n_regimes=3)
    states = gmm_det.fit_predict(variance_series)

    assert len(states) == len(variance_series)
    assert set(states.unique()) == {0, 1, 2}


def test_regime_analyzer_transition_matrix(variance_series):
    detector = StatisticalRegimeDetector(warmup_period=200)
    regimes = detector.detect(variance_series)

    t_matrix = RegimeAnalyzer.compute_transition_matrix(regimes)
    assert not t_matrix.empty

    # Each row of transition probability matrix must sum to 1.0
    row_sums = t_matrix.sum(axis=1)
    for s in row_sums:
        assert np.isclose(s, 1.0, atol=1e-5)

    durations = RegimeAnalyzer.compute_expected_durations(t_matrix)
    assert len(durations) == len(t_matrix)
    for dur in durations.values():
        assert dur >= 1.0
