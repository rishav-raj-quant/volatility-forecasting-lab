# Volatility Forecasting Lab: Research Methodology

## 1. Research Philosophy

Financial market volatility is unobservable, latent, and stochastic. Unlike price or return, which can be directly recorded at discrete points in time, volatility must be filtered, estimated, or inferred from market observations.

The primary objective of the **Volatility Forecasting Lab** is to build a mathematically rigorous, reproducible, out-of-sample validated quantitative research and risk management platform. Rather than engineering arbitrary features to maximize in-sample fit ($R^2$), the platform prioritizes:
1. **Economic & Statistical Rationale**: Every feature and model is rooted in verified financial econometrics.
2. **Leakage-Safe Validation**: Absolute temporal causality is maintained throughout data ingestion, feature generation, normalization, and model evaluation.
3. **Robust Loss Functions**: Utilizing Patton (2011) Quasi-Likelihood (QLIKE) loss, which is mathematically robust to noise in realized variance proxies.

---

## 2. Multi-Horizon Forecasting Setup

The platform models volatility across four primary institutional horizons:
- **$h = 1$ trading day**: Intraday / overnight risk management, daily portfolio rebalancing.
- **$h = 5$ trading days (1 week)**: Short-term tactical allocation, options delta-hedging.
- **$h = 10$ trading days (2 weeks)**: Regulatory Basel backtesting window.
- **$h = 20$ trading days (~1 month)**: Standard monthly options expiration cycle and strategic risk budgeting.

For any horizon $h \ge 1$, the target is defined as the forward average daily realized variance:
$$y_t^{(h)} = \frac{1}{h} \sum_{k=1}^h r_{t+k}^2$$

The corresponding annualized volatility target in decimal units is:
$$\sigma_{t+1:t+h}^{\text{ann}} = \sqrt{252 \times y_t^{(h)}}$$

---

## 3. Walk-Forward Protocol

Standard K-fold cross-validation is strictly forbidden in quantitative time-series modeling because shuffling observations leaks future volatility regimes into historical folds.

The platform enforces:
- **Expanding Window Validation**: The training window starts at $t=0$ and expands chronologically, mimicking real-world institutional forecasting.
- **Purging**: When forecasting over horizon $h > 1$, training and test sets are separated by a purge window of $h - 1$ days to prevent label overlap.
- **Embargoing**: Additional buffer days are placed after the test set to account for auto-correlated feature lags.
