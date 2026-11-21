# Data Dictionary

## 1. Raw & Processed Market Data

| Column | Type | Units | Description |
|---|---|---|---|
| **Date** | DatetimeIndex | UTC / Asia/Kolkata | Market session trading date. |
| **Open** | float64 | Currency (INR/USD) | Session opening price. |
| **High** | float64 | Currency (INR/USD) | Session highest traded price. |
| **Low** | float64 | Currency (INR/USD) | Session lowest traded price. |
| **Close** | float64 | Currency (INR/USD) | Session official closing price. |
| **Adj Close** | float64 | Currency (INR/USD) | Split and dividend-adjusted closing price. |
| **Volume** | float64 | Shares / Contracts | Total volume transacted during session. |
| **log_return** | float64 | Dimensionless (decimal) | Daily continuous return: $\ln(P_t / P_{t-1})$. |
| **simple_return** | float64 | Dimensionless (decimal) | Daily percentage change: $(P_t - P_{t-1}) / P_{t-1}$. |
| **overnight_return**| float64 | Dimensionless (decimal) | Return from previous close to open: $\ln(O_t / C_{t-1})$. |
| **intraday_return** | float64 | Dimensionless (decimal) | Session return: $\ln(C_t / O_t)$. |
| **daily_variance_proxy**| float64 | Variance (decimal^2) | Unbiased daily squared return proxy: $r_t^2$. |

---

## 2. Engineered Feature Columns

| Feature Name | Category | Description | Formula / Specification |
|---|---|---|---|
| `ret_lag_1` ... `ret_lag_20` | Returns | Backward-looking lagged log returns | $r_{t - \text{lag} + 1}$ |
| `abs_ret_lag_1` ... | Returns | Magnitude / absolute returns | $\|r_{t - \text{lag} + 1}\|$ |
| `sq_ret_lag_1` ... | Returns | Squared returns (variance shocks) | $r_{t - \text{lag} + 1}^2$ |
| `hist_vol_5d` ... `hist_vol_60d` | Estimator | Rolling sample standard deviation | Rolling $\sigma_{CC}$ over window |
| `ewma_vol_94`, `ewma_vol_97` | Estimator | RiskMetrics EWMA volatility | Recursive decay $\lambda \in \{0.94, 0.97\}$ |
| `parkinson_20d` | Estimator | Extreme value High-Low range vol | Parkinson (1980) formula |
| `garman_klass_20d` | Estimator | OHLC jump-adjusted range vol | Garman-Klass (1980) formula |
| `rogers_satchell_20d` | Estimator | Drift-independent range vol | Rogers-Satchell (1991) formula |
| `rsi_14` | Technical | Relative Strength Index | 14-day Wilder RSI ($0 - 100$) |
| `atr_norm_14` | Technical | Normalized Average True Range | ATR(14) / Close |
| `dist_sma_20` ... | Technical | Moving average distance | $(P_t - \text{SMA}_k) / \text{SMA}_k$ |
| `drawdown_20d` ... | Technical | Rolling peak-to-trough loss | $(P_t - \max_{k} P_k) / \max_{k} P_k$ |
| `volume_zscore_20d` | Volume | 20-day normalized volume z-score | $(V_t - \bar{V}_{20}) / \sigma(V_{20})$ |
