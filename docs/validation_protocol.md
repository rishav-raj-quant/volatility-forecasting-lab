# Validation Protocol & Leakage Prevention

## 1. Principles of Financial Time-Series Validation

In standard machine learning, datasets are frequently shuffled and split randomly using $k$-fold cross-validation. For financial markets, **random cross-validation is fatal** because:
1. Volatility is auto-correlated; an observation on day $t$ contains immediate information about day $t-1$ and $t+1$.
2. Shuffling creates lookahead bias, inflating backtest scores with zero predictive validity on future data.

---

## 2. Walk-Forward Expanding Windows

The **Volatility Forecasting Lab** implements strict forward chronological evaluation:

```
Fold 1: [--- Train: 2018-2020 ---] [Purge] [Test: Q1 2021]
Fold 2: [------ Train: 2018-Q1 2021 ------] [Purge] [Test: Q2 2021]
Fold 3: [--------- Train: 2018-Q2 2021 ---------] [Purge] [Test: Q3 2021]
...
```

### Key Enforcements:
- **No Future Scaler Parameters**: Mean and standard deviation are calculated solely on the training fold $X_{\text{train}}$ and then applied to transform $X_{\text{test}}$.
- **Purge Window**: When forecasting horizon $h > 1$, a purge window of $h - 1$ trading sessions is discarded immediately preceding the test window to prevent forward-looking label overlap.
- **Embargo Buffer**: An optional embargo period handles post-test autocorrelation.
- **Unbroken Temporal Ordering**: At no point in the training pipeline does any observation at time $\tau > t$ enter the feature set.
