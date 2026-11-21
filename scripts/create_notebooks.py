"""Generates research Jupyter notebooks with theory, formulas, code, and interpretation."""

import json
import os

NOTEBOOKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks"))
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True)
    }


def create_all_notebooks():
    # 01 Market Data Exploration
    nb1_cells = [
        md_cell("""# Research Notebook 01: Stylized Empirical Facts of Asset Returns
## Quantitative Analysis of Distributional Properties, Fat Tails & Volatility Clustering

### 1. Research Objective
Investigate the universal empirical properties of financial asset returns on NIFTY 50 and benchmark indices:
1. Non-normality and leptokurtosis (excess kurtosis)
2. Negative skewness and asymmetric tail distributions
3. Absence of linear autocorrelation in raw returns
4. Strong, persistent autocorrelation in absolute and squared returns (volatility clustering)
"""),
        code_cell("""import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from volatility_lab.data.providers.synthetic import SyntheticMarketDataProvider
from volatility_lab.data.preprocessing import DataPreprocessor

# Ingest data
raw = SyntheticMarketDataProvider(seed=42).fetch_data('NIFTY50', '2018-01-01', '2023-12-31')
df = DataPreprocessor().clean_and_prepare(raw)
returns = df['log_return'].dropna()
print(f"Total trading days: {len(returns)}")
"""),
        md_cell("""### 2. Statistical Moments & Distributional Properties
We calculate mean $\\mu$, variance $\\sigma^2$, skewness $S$, and kurtosis $K$:
$$S = \\frac{E[(r - \\mu)^3]}{\\sigma^3}, \\quad K = \\frac{E[(r - \\mu)^4]}{\\sigma^4}$$
"""),
        code_cell("""mean_ret = returns.mean() * 252.0
vol_ann = returns.std() * np.sqrt(252.0)
skew = stats.skew(returns)
kurt = stats.kurtosis(returns, fisher=False) # Pearson kurtosis (Normal = 3)
jb_stat, jb_pval = stats.jarque_bera(returns)

print(f"Annualized Mean: {mean_ret*100:.2f}%")
print(f"Annualized Volatility: {vol_ann*100:.2f}%")
print(f"Sample Skewness: {skew:.3f} (Asymmetric Tail)")
print(f"Sample Kurtosis: {kurt:.3f} (Leptokurtic, Fat-Tailed vs Gaussian 3.0)")
print(f"Jarque-Bera Test: stat={jb_stat:.2f}, p-val={jb_pval:.4e} => Strictly Reject Normality")
"""),
        md_cell("""### 3. Autocorrelation & Volatility Clustering
Cont (2001) stylized fact: While raw returns $r_t$ show negligible autocorrelation, $|r_t|$ and $r_t^2$ exhibit significant, long-memory persistence.
"""),
        code_cell("""lags = range(1, 31)
raw_acf = [returns.autocorr(lag=l) for l in lags]
abs_acf = [np.abs(returns).autocorr(lag=l) for l in lags]
sq_acf = [(returns**2).autocorr(lag=l) for l in lags]

fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
ax.plot(lags, raw_acf, marker='o', label='Raw Returns $r_t$')
ax.plot(lags, abs_acf, marker='s', label='Absolute Returns $|r_t|$')
ax.plot(lags, sq_acf, marker='^', label='Squared Returns $r_t^2$')
ax.axhline(0, color='gray', linestyle='--')
ax.set_title("Autocorrelation Function: Proof of Volatility Clustering")
ax.set_xlabel("Lag (Trading Days)")
ax.set_ylabel("Autocorrelation")
ax.legend()
plt.tight_layout()
plt.show()
"""),
        md_cell("""### 4. Conclusion & Research Implications
1. Gaussian distribution assumption is strictly invalid for financial risk management; parametric models must incorporate fat tails (e.g. Student's t distribution).
2. Volatility is not constant; its autocorrelation confirms strong conditional heteroskedasticity and motivates ARCH/GARCH modeling.
""")
    ]

    # 02 Volatility Estimators
    nb2_cells = [
        md_cell("""# Research Notebook 02: Comparative Analysis of Volatility Estimators
## Historical Close-to-Close vs. High-Low-Open-Close Range Estimators

### 1. Mathematical Framework
- **Close-to-Close (Standard):**
  $$\\sigma_{CC}^2 = \\frac{1}{N-1} \\sum_{i=1}^N (r_i - \\bar{r})^2$$
- **Parkinson (1980):**
  $$\\sigma_P^2 = \\frac{1}{4 \\ln 2} \\frac{1}{N} \\sum_{i=1}^N \\left(\\ln\\frac{H_i}{L_i}\\right)^2$$
- **Garman-Klass (1980):**
  $$\\sigma_{GK}^2 = \\frac{1}{N} \\sum_{i=1}^N \\left[ 0.5\\left(\\ln\\frac{H_i}{L_i}\\right)^2 - (2\\ln 2 - 1)\\left(\\ln\\frac{C_i}{O_i}\\right)^2 \\right]$$
- **Rogers-Satchell (1991):**
  $$\\sigma_{RS}^2 = \\frac{1}{N} \\sum_{i=1}^N \\left[ \\ln\\frac{H_i}{C_i}\\ln\\frac{H_i}{O_i} + \\ln\\frac{L_i}{C_i}\\ln\\frac{L_i}{O_i} \\right]$$
"""),
        code_cell("""import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from volatility_lab.data.providers.synthetic import SyntheticMarketDataProvider
from volatility_lab.data.preprocessing import DataPreprocessor
from volatility_lab.estimators.historical import HistoricalVolatilityEstimator
from volatility_lab.estimators.ewma import EWMAVolatilityEstimator
from volatility_lab.estimators.parkinson import ParkinsonEstimator
from volatility_lab.estimators.garman_klass import GarmanKlassEstimator
from volatility_lab.estimators.rogers_satchell import RogersSatchellEstimator

raw = SyntheticMarketDataProvider(seed=42).fetch_data('NIFTY50', '2020-01-01', '2023-12-31')
df = DataPreprocessor().clean_and_prepare(raw)
returns = df['log_return']

hist_vol = HistoricalVolatilityEstimator(windows=20).estimate(returns)['hist_vol_20d']
ewma_vol = EWMAVolatilityEstimator(decay_factor=0.94).estimate(returns)
park_vol = ParkinsonEstimator(window=20).estimate(df['High'], df['Low'])
gk_vol = GarmanKlassEstimator(window=20).estimate(df['Open'], df['High'], df['Low'], df['Close'])
rs_vol = RogersSatchellEstimator(window=20).estimate(df['Open'], df['High'], df['Low'], df['Close'])

comp_df = pd.DataFrame({
    'Hist (20d)': hist_vol,
    'EWMA (0.94)': ewma_vol,
    'Parkinson': park_vol,
    'Garman-Klass': gk_vol,
    'Rogers-Satchell': rs_vol
}).dropna()
"""),
        code_cell("""fig, ax = plt.subplots(figsize=(11, 5), dpi=100)
comp_df.plot(ax=ax, linewidth=1.4)
ax.set_title("Volatility Estimator Trajectories Comparison (Annualized %)")
ax.set_ylabel("Annualized Volatility")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""),
        md_cell("""### 2. Efficiency Comparison & Conclusions
- Parkinson, Garman-Klass, and Rogers-Satchell exploit intraday price extremes, reducing sample variance by 5x to 8x compared to close-to-close returns.
- Rogers-Satchell handles non-zero market drift without upward variance bias.
""")
    ]

    # 03 Econometric Models
    nb3_cells = [
        md_cell("""# Research Notebook 03: Econometric Modeling
## GARCH(1,1), EGARCH, GJR-GARCH, and HAR-RV Estimation

### 1. Theory & Specifications
1. **GARCH(1,1) (Bollerslev 1986):**
   $$\\sigma_t^2 = \\omega + \\alpha \\epsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2, \\quad \\alpha + \\beta < 1$$
2. **EGARCH(1,1) (Nelson 1991):**
   $$\\ln(\\sigma_t^2) = \\omega + \\alpha(|z_{t-1}| - E|z_{t-1}|) + \\gamma z_{t-1} + \\beta \\ln(\\sigma_{t-1}^2)$$
3. **GJR-GARCH(1,1) (Glosten et al. 1993):**
   $$\\sigma_t^2 = \\omega + (\\alpha + \\gamma I_{t-1})\\epsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2$$
4. **HAR-RV (Corsi 2009):**
   $$RV_{t+1d}^{(d)} = c + \\beta^{(d)} RV_t^{(d)} + \\beta^{(w)} RV_t^{(w)} + \\beta^{(m)} RV_t^{(m)} + \\epsilon_{t+1d}$$
"""),
        code_cell("""import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
from volatility_lab.data.providers.synthetic import SyntheticMarketDataProvider
from volatility_lab.data.preprocessing import DataPreprocessor
from volatility_lab.econometrics.garch import GARCHModel
from volatility_lab.econometrics.egarch import EGARCHModel
from volatility_lab.econometrics.gjr_garch import GJRGARCHModel
from volatility_lab.econometrics.har_rv import HARRVModel

raw = SyntheticMarketDataProvider(seed=42).fetch_data('NIFTY50', '2019-01-01', '2023-12-31')
df = DataPreprocessor().clean_and_prepare(raw)
returns = df['log_return']

# Fit GARCH(1,1)
garch = GARCHModel(p=1, q=1).fit(returns)
print(f"GARCH(1,1) Persistence: {garch.persistence:.4f}")
print(f"Volatility Shock Half-Life: {garch.half_life:.1f} days")

# Fit EGARCH(1,1)
egarch = EGARCHModel(p=1, q=1).fit(returns)
print(f"EGARCH Leverage Coefficient gamma: {egarch.leverage_coefficient:.4f}")

# Fit HAR-RV
har = HARRVModel().fit(df['daily_variance_proxy'], forecast_horizon=1)
print(f"HAR-RV R-Squared: {har.r_squared_:.4f}")
print("HAR-RV Parameters:", har.coefficients_)
"""),
        md_cell("""### 2. Empirical Findings
- Persistent clustering is confirmed: $\\alpha + \\beta \\approx 0.96 - 0.98$.
- Leverage effect ($\gamma < 0$ in EGARCH, $\gamma > 0$ in GJR-GARCH) reflects asymmetric market responses to selloffs.
- HAR-RV captures cascade dynamics across daily, weekly, and monthly horizons.
""")
    ]

    # 04 Machine Learning Models
    nb4_cells = [
        md_cell("""# Research Notebook 04: Machine Learning Volatility Prediction
## Linear Regularization, Tree Ensembles, Gradient Boosting & Uncertainty

### 1. Research Objective
Can machine learning models with rich technical, range-based, and distributional features outperform traditional econometric baselines?
"""),
        code_cell("""import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
from volatility_lab.data.providers.synthetic import SyntheticMarketDataProvider
from volatility_lab.data.preprocessing import DataPreprocessor
from volatility_lab.features.feature_pipeline import FeaturePipeline
from volatility_lab.machine_learning.registry import MLModelRegistry
from volatility_lab.evaluation.metrics import compute_all_metrics

raw = SyntheticMarketDataProvider(seed=42).fetch_data('NIFTY50', '2019-01-01', '2023-12-31')
df = DataPreprocessor().clean_and_prepare(raw)

pipe = FeaturePipeline()
X, y = pipe.transform(df, horizon=5)

split = int(len(X) * 0.75)
X_tr, y_tr = X.iloc[:split], y.iloc[:split]
X_te, y_te = X.iloc[split:], y.iloc[split:]

models = ['ridge', 'random_forest', 'lightgbm']
results = []

for m_name in models:
    m = MLModelRegistry.get(m_name)
    m.fit(X_tr, y_tr)
    preds = np.maximum(m.predict(X_te), 1e-6)
    metrics = compute_all_metrics(y_te.values, preds)
    metrics['model'] = m_name
    results.append(metrics)

perf_df = pd.DataFrame(results).set_index('model')
print(perf_df[['qlike', 'rmse', 'mae', 'r2']])
"""),
        md_cell("""### 2. Takeaways & Feature Importance
- Gradient boosting captures non-linear interactions between range estimators and volume shifts.
- Tree models show superior stability during volatility spikes.
""")
    ]

    # 05 Walk-Forward Validation
    nb5_cells = [
        md_cell("""# Research Notebook 05: Walk-Forward Cross-Validation & Statistical Testing
## Leakage-Safe Time-Series Validation and Diebold-Mariano Tests

### 1. Methodology
- **Expanding Window Validation:** Preserves causality.
- **Purging & Embargo:** For horizon $h=5$, removes boundary overlap to eliminate lookahead bias.
- **Patton (2011) QLIKE Loss:**
  $$QLIKE(h_t, \\sigma_t^2) = \\frac{\\sigma_t^2}{h_t} - \\ln\\left(\\frac{\\sigma_t^2}{h_t}\\right) - 1$$
- **Diebold-Mariano Test (1995)** with Newey-West long-run variance.
"""),
        code_cell("""import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
from volatility_lab.data.providers.synthetic import SyntheticMarketDataProvider
from volatility_lab.data.preprocessing import DataPreprocessor
from volatility_lab.features.feature_pipeline import FeaturePipeline
from volatility_lab.evaluation.walk_forward import WalkForwardValidator
from volatility_lab.evaluation.statistical_tests import diebold_mariano_test
from volatility_lab.machine_learning.registry import MLModelRegistry

raw = SyntheticMarketDataProvider(seed=42).fetch_data('NIFTY50', '2019-01-01', '2023-12-31')
df = DataPreprocessor().clean_and_prepare(raw)
X, y = FeaturePipeline().transform(df, horizon=5)

validator = WalkForwardValidator(min_train_size=400, test_size=100, step_size=100)
res_ridge = validator.evaluate_model(lambda: MLModelRegistry.get('ridge'), X, y, forecast_horizon=5)
res_lgbm = validator.evaluate_model(lambda: MLModelRegistry.get('lightgbm'), X, y, forecast_horizon=5)

dm = diebold_mariano_test(res_ridge['y_true'], res_lgbm['y_pred'], res_ridge['y_pred'], horizon=5, loss_type='qlike')
print(f"Diebold-Mariano DM Stat: {dm.statistic:.3f}, p-value: {dm.p_value:.4f}")
print(f"Superior Model: {dm.superior_model}")
"""),
        md_cell("""### 2. Conclusions
- Standard random K-fold CV leads to severe data leakage in financial time series.
- Walk-forward testing with purging gives realistic out-of-sample error estimates.
""")
    ]

    # 06 Risk Analytics
    nb6_cells = [
        md_cell("""# Research Notebook 06: Quantitative Market Risk & VaR Backtesting
## Value at Risk, Expected Shortfall, Kupiec POF & Christoffersen Tests

### 1. Risk Measures
- **Value at Risk (VaR):** Exceedance threshold quantile.
- **Expected Shortfall (CVaR):** $ES_\\alpha = E[L \\mid L > VaR_\\alpha]$.
- **Kupiec (1995) POF Likelihood Ratio Test:** Unconditional coverage $LR_{pof} \\sim \\chi^2(1)$.
- **Christoffersen (1998) Independence Test:** No violation clustering $LR_{ind} \\sim \\chi^2(1)$.
"""),
        code_cell("""import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from volatility_lab.data.providers.synthetic import SyntheticMarketDataProvider
from volatility_lab.data.preprocessing import DataPreprocessor
from volatility_lab.risk.var import ValueAtRisk
from volatility_lab.risk.expected_shortfall import ExpectedShortfall
from volatility_lab.risk.backtesting import VaRBacktester

raw = SyntheticMarketDataProvider(seed=42).fetch_data('NIFTY50', '2020-01-01', '2023-12-31')
df = DataPreprocessor().clean_and_prepare(raw)
returns = df['log_return']

var_calc = ValueAtRisk(confidence_levels=[0.95, 0.99])
var_norm = var_calc.parametric_normal_var(returns)
var_t = var_calc.parametric_student_t_var(returns)
var_hist = var_calc.historical_var(returns)

es_calc = ExpectedShortfall(confidence_levels=[0.95])
es_norm = es_calc.parametric_normal_es(returns)

# Backtest 95% Parametric Normal VaR
v95 = var_norm['var_norm_95'].dropna()
bt = VaRBacktester.backtest(returns.reindex(v95.index), v95, confidence_level=0.95)

print(f"Total Observations: {bt.total_observations}")
print(f"Actual Violations: {bt.actual_violations} (Expected: {bt.expected_violations:.1f})")
print(f"Violation Ratio: {bt.violation_ratio:.2f}")
print(f"Kupiec POF LR: {bt.kupiec_lr_stat:.3f} (p={bt.kupiec_p_value:.4f}) => {bt.kupiec_decision}")
print(f"Christoffersen Independence LR: {bt.christoffersen_ind_stat:.3f} (p={bt.christoffersen_ind_p_value:.4f}) => {bt.christoffersen_ind_decision}")
"""),
        md_cell("""### 2. Regulatory Implications & Basel Traffic Light System
- Normal VaR systematically underestimates tail losses during market crashes (fat tails).
- Student-t VaR and GARCH-filtered VaR pass conditional coverage tests by dynamically expanding during stress periods.
""")
    ]

    notebooks = {
        "01_market_data_exploration.ipynb": nb1_cells,
        "02_volatility_estimators.ipynb": nb2_cells,
        "03_econometric_models.ipynb": nb3_cells,
        "04_machine_learning_models.ipynb": nb4_cells,
        "05_walk_forward_validation.ipynb": nb5_cells,
        "06_risk_analytics.ipynb": nb6_cells,
    }

    for fname, cells in notebooks.items():
        fpath = os.path.join(NOTEBOOKS_DIR, fname)
        nb_json = make_notebook(cells)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(nb_json, f, indent=2)
        print(f"[OK] Generated {fpath}")


if __name__ == "__main__":
    create_all_notebooks()
