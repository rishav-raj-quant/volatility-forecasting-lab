"""Unit tests for deep learning sequential volatility architectures."""

import pytest
import numpy as np
import pandas as pd
from volatility_lab.deep_learning.sequential import DeepVolatilityForecaster, TORCH_AVAILABLE


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for deep learning tests.")
@pytest.mark.parametrize("arch", ["lstm", "gru", "tcn", "attention"])
def test_deep_architectures(arch):
    np.random.seed(42)
    n = 100
    p = 5
    X = pd.DataFrame(np.random.normal(0, 1, (n, p)), columns=[f"feat_{i}" for i in range(p)])
    y = pd.Series(np.random.exponential(0.04, n))

    model = DeepVolatilityForecaster(
        architecture=arch,
        seq_length=10,
        hidden_dim=16,
        epochs=3,
        batch_size=16,
        patience=2,
    )
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) > 0
    assert (preds > 0.0).all()
