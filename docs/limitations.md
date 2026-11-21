# Known Limitations & Research Boundaries

The **Volatility Forecasting Lab** is built for research, risk modeling, and derivative valuation. It is essential to recognize the real-world operational and econometric constraints of the system.

---

## 1. Econometric & Data Limitations

1. **Proxy Noise**: Daily squared returns $r_t^2$ are unbiased proxies for daily integrated variance, but they contain substantial idiosyncratic noise. While Patton (2011) QLIKE loss mitigates ranking distortion, intraday high-frequency sampling (e.g. 5-minute realized variance) provides higher signal-to-noise ratios when available.
2. **Trading Halts & Illiquidity**: Assets with frequent zero-volume sessions, limit-up/limit-down trading halts, or wide bid-ask spreads violate continuous-diffusion assumptions and can distort range estimators (Parkinson, Garman-Klass).
3. **Overnight Gap Risk**: Range estimators that ignore opening jumps underestimate total variance in markets with heavy overnight corporate or geopolitical news flow.

---

## 2. Model & Market Risk Boundaries

1. **Stationarity Assumptions**: Econometric models (GARCH) assume structural parameters remain stationary over the sample. Extreme regime shifts (e.g., March 2020 COVID shock) cause transient parameter breakdown until rolling windows adapt.
2. **VaR is Not Maximum Loss**: Value at Risk represents an exceedance quantile, not the ceiling of potential loss. Expected Shortfall (CVaR) must always be examined alongside VaR to understand tail-loss severity.
3. **Not an Execution Strategy**: Volatility forecasts provide risk inputs and pricing parameters, not standalone alpha or directional trading signals.
