# Mathematical Framework & Econometric Formulations

This document details the mathematical specifications for all estimators, models, loss functions, and statistical tests implemented in the **Volatility Forecasting Lab**.

---

## 1. Return Formulations

Given closing prices $P_t$ and $P_{t-1}$, log returns are defined as:
$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

Overnight returns:
$$r_{\text{overnight}, t} = \ln\left(\frac{\text{Open}_t}{\text{Close}_{t-1}}\right)$$

Intraday open-to-close returns:
$$r_{\text{intraday}, t} = \ln\left(\frac{\text{Close}_t}{\text{Open}_t}\right)$$

---

## 2. Volatility Estimators

### Historical Close-to-Close Volatility
$$\sigma_{CC} = \sqrt{\frac{252}{N-1} \sum_{i=1}^N (r_i - \bar{r})^2}$$

### Exponentially Weighted Moving Average (EWMA / RiskMetrics)
$$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) r_{t-1}^2, \quad \lambda = 0.94$$

### Parkinson (1980) High-Low Range Estimator
Exploits continuous price path extremes between daily high $H_i$ and low $L_i$:
$$\sigma_P^2 = \frac{1}{4 \ln 2} \frac{1}{N} \sum_{i=1}^N \left(\ln\frac{H_i}{L_i}\right)^2$$
Efficiency is approximately $5\times$ higher than close-to-close volatility.

### Garman-Klass (1980) OHLC Estimator
Incorporates opening jumps and intraday trading range:
$$\sigma_{GK}^2 = \frac{1}{N} \sum_{i=1}^N \left[ 0.5 \left(\ln\frac{H_i}{L_i}\right)^2 - (2\ln 2 - 1) \left(\ln\frac{C_i}{O_i}\right)^2 \right]$$
Efficiency is up to $8\times$ higher than standard close-to-close volatility.

### Rogers-Satchell (1991) Estimator
Invariant to non-zero drift $\mu \neq 0$:
$$\sigma_{RS}^2 = \frac{1}{N} \sum_{i=1}^N \left[ \ln\left(\frac{H_i}{C_i}\right)\ln\left(\frac{H_i}{O_i}\right) + \ln\left(\frac{L_i}{C_i}\right)\ln\left(\frac{L_i}{O_i}\right) \right]$$

### Realized Variance & Bipower Variation
Given $M$ intraday returns $r_{t,j}$:
$$RV_t = \sum_{j=1}^M r_{t,j}^2$$
$$BV_t = \frac{\pi}{2} \left(\frac{M}{M-1}\right) \sum_{j=2}^M |r_{t,j}| |r_{t,j-1}|$$
Jump variation: $J_t = \max(RV_t - BV_t, 0)$.

---

## 3. Econometric Volatility Models

### GARCH(1,1) (Bollerslev 1986)
$$r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim \text{i.i.d.}(0, 1)$$
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
Stationarity condition: $\alpha + \beta < 1$.
Unconditional variance: $\bar{\sigma}^2 = \frac{\omega}{1 - \alpha - \beta}$.
Shock half-life: $\tau = \frac{\ln(0.5)}{\ln(\alpha + \beta)}$ trading days.

### EGARCH(1,1) (Nelson 1991)
$$\ln(\sigma_t^2) = \omega + \alpha \left(|z_{t-1}| - \mathbb{E}|z_{t-1}|\right) + \gamma z_{t-1} + \beta \ln(\sigma_{t-1}^2)$$
where $\gamma < 0$ captures the leverage effect (negative returns increase volatility more than positive returns).

### GJR-GARCH(1,1) (Glosten, Jagannathan, Runkle 1993)
$$\sigma_t^2 = \omega + (\alpha + \gamma I_{t-1}) \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
$$I_{t-1} = \begin{cases} 1 & \text{if } \epsilon_{t-1} < 0 \\ 0 & \text{otherwise} \end{cases}$$

### HAR-RV (Corsi 2009)
$$RV_{t+h} = \beta_0 + \beta_d RV_t^{(d)} + \beta_w RV_t^{(w)} + \beta_m RV_t^{(m)} + \epsilon_{t+h}$$
where:
$$RV_t^{(d)} = RV_t, \quad RV_t^{(w)} = \frac{1}{5}\sum_{i=0}^4 RV_{t-i}, \quad RV_t^{(m)} = \frac{1}{22}\sum_{i=0}^{21} RV_{t-i}$$

---

## 4. Evaluation Loss Functions

### Patton (2011) Quasi-Likelihood (QLIKE)
$$QLIKE(h_t, \sigma_t^2) = \frac{\sigma_t^2}{h_t} - \ln\left(\frac{\sigma_t^2}{h_t}\right) - 1$$
where $\sigma_t^2$ is the realized variance proxy and $h_t$ is the model variance forecast.
Properties: Strictly robust to noisy, imperfect proxies; scale-invariant; penalizes under-forecasting more heavily than over-forecasting.

---

## 5. Statistical Hypothesis Testing

### Diebold-Mariano Test (1995)
Differential loss: $d_t = L(e_{1,t}) - L(e_{2,t})$.
$$\bar{d} = \frac{1}{T} \sum_{t=1}^T d_t$$
$$DM = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d}) / T}} \xrightarrow{d} \mathcal{N}(0, 1)$$
where $\hat{V}(\bar{d})$ is estimated via the Newey-West Bartlett kernel HAC estimator with lag $k = h - 1$:
$$\hat{V}(\bar{d}) = \hat{\gamma}_0 + 2 \sum_{k=1}^{h-1} \left(1 - \frac{k}{h}\right) \hat{\gamma}_k$$

---

## 6. Quantitative Risk & Regulatory Backtesting

### Kupiec (1995) Proportion of Failures (POF) LR Test
$$LR_{pof} = -2 \ln \left[ \frac{(1-p)^{N-x} p^x}{(1 - \hat{p})^{N-x} \hat{p}^x} \right] \sim \chi^2(1)$$
where $N$ is sample size, $x$ is actual violations, $p = 1 - \alpha$, and $\hat{p} = x / N$.

### Christoffersen (1998) Independence Test
Tests whether exceedances are clustered ($I_t \to I_{t+1}$ transition):
$$LR_{ind} = -2 \ln \left[ \frac{\pi_0^{n_{00}+n_{10}} (1-\pi_0)^{n_{01}+n_{11}}}{\pi_{01}^{n_{01}} (1-\pi_{01})^{n_{00}} \pi_{11}^{n_{11}} (1-\pi_{11})^{n_{10}}} \right] \sim \chi^2(1)$$

### Conditional Coverage Joint Test
$$LR_{cc} = LR_{pof} + LR_{ind} \sim \chi^2(2)$$
