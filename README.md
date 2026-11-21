# Volatility Forecasting Lab

A quantitative research and financial machine learning platform for forecasting market volatility using econometric models, statistical learning, realized volatility estimation, walk-forward validation, and quantitative risk analytics.

---

## Key Capabilities

1. **Volatility Estimation**:
   - Historical Close-to-Close standard deviation (annualized by $\sqrt{252}$).
   - RiskMetrics Exponentially Weighted Moving Average (EWMA).
   - Range-based estimators: Parkinson High-Low, Garman-Klass OHLC, and Rogers-Satchell drift-independent formulations.
   - High-frequency realized variance, bipower variation, and jump detection.

2. **Econometric Modeling**:
   - ARCH, GARCH(1,1) with Gaussian and Student-t innovations.
   - Exponential GARCH (EGARCH) capturing asymmetric leverage effects.
   - Glosten-Jagannathan-Runkle GJR-GARCH threshold asymmetry.
   - Corsi Heterogeneous Autoregressive Realized Volatility (HAR-RV).

3. **Machine Learning & Deep Learning**:
   - Regularized linear regressors: Ridge, Lasso, ElasticNet.
   - Ensembles: Random Forest, Extra Trees, Gradient Boosting, LightGBM, and XGBoost.
   - Non-linear Support Vector Regression (SVR).
   - Quantile regression for predictive uncertainty intervals (10th, 50th, 90th percentiles).
   - Sequential deep learning: PyTorch LSTM, GRU, Temporal Convolutional Network (TCN), and Self-Attention.

4. **Multi-Horizon Forecasting**:
   - 1-day, 5-day, 10-day, and 20-day horizon forward targets.
   - Fan charts displaying forecast trajectories and 95% predictive cones.

5. **Validation & Regulatory Risk Analytics**:
   - Strict walk-forward cross-validation with purging and embargoing.
   - Model comparison via Patton Quasi-Likelihood (QLIKE) loss, RMSE, MAE, and $R^2$.
   - Pairwise Diebold-Mariano hypothesis testing with Newey-West long-run variance adjustment.
   - Value at Risk (Historical, Parametric Normal, Student-t, GARCH).
   - Expected Shortfall (Conditional VaR).
   - Basel regulatory backtesting: Kupiec POF test and Christoffersen independence test.

---

## Architecture Overview

```
volatility-forecasting-lab/
├── app/                  # Streamlit Interactive Research Workstation
│   └── streamlit_app.py
├── config/               # Settings & Model Hyperparameters
│   ├── settings.yaml
│   └── model_config.yaml
├── data/                 # Data partitions (raw, processed, features, predictions)
├── docs/                 # Mathematical framework, methodology, validation protocols
├── notebooks/            # 6 quantitative research notebooks
├── scripts/              # Ingestion, experiment execution, report generation
├── src/volatility_lab/   # Core analytical package
│   ├── data/             # Ingestion, validation, preprocessing, storage
│   ├── estimators/       # Historical, EWMA, Parkinson, Garman-Klass, Rogers-Satchell, Realized
│   ├── econometrics/     # ARCH, GARCH, EGARCH, GJR-GARCH, HAR-RV
│   ├── machine_learning/ # Ridge, Lasso, ElasticNet, RF, GBDT, LightGBM, XGBoost, SVR, Quantile
│   ├── deep_learning/    # PyTorch sequential models (LSTM, GRU, TCN, Attention)
│   ├── forecasting/      # Multi-horizon target construction, uncertainty cones
│   ├── evaluation/       # QLIKE loss, Diebold-Mariano tests, walk-forward validator
│   ├── regimes/          # Statistical thresholds, GMM, Markov transition probabilities
│   ├── risk/             # VaR, Expected Shortfall, Kupiec & Christoffersen backtesting
│   ├── experiments/      # Experiment tracker, leaderboards, report generation
│   ├── visualization/    # Interactive Plotly financial charts
│   └── cli.py            # Command-Line Interface
└── tests/                # Comprehensive unit and integration test suite (49 tests)
```

---

## Quickstart & Installation

```bash
git clone https://github.com/rishav-raj-quant/volatility-forecasting-lab.git
cd volatility-forecasting-lab

pip install -r requirements.txt
pip install -e .
```

### Running the Interactive Dashboard

```bash
streamlit run app/streamlit_app.py
```

### CLI Commands

```bash
# Ingest and validate market data
python -m volatility_lab.cli download --ticker ^NSEI

# Compare volatility estimators
python -m volatility_lab.cli estimate --ticker ^NSEI

# Generate multi-model forecasts for a 5-day horizon
python -m volatility_lab.cli forecast --ticker ^NSEI --horizon 5

# Compute Value at Risk, Expected Shortfall, and regulatory backtests
python -m volatility_lab.cli risk --ticker ^NSEI

# Generate formal research report in Markdown
python -m volatility_lab.cli report --ticker ^NSEI --output experiments/research_report.md
```

### Automated Testing

Run the full pytest suite:

```bash
pytest -v
```

All 49 tests pass across unit calculations, econometric models, and end-to-end pipelines.
