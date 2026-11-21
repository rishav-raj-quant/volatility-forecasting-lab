# Model Documentation & Catalog

This catalog outlines all econometric, machine learning, and deep learning models supported in the **Volatility Forecasting Lab**.

## 1. Econometric Models

| Model | Class | Target | Key Hyperparameters | Theoretical Advantage |
|---|---|---|---|---|
| **ARCH(p)** | Econometric | Variance | `p` (lags: 1, 2, 5) | Foundational autoregressive conditional variance. |
| **GARCH(1,1)** | Econometric | Variance | `p=1, q=1, dist='normal'/'t'` | Parsimonious persistence model, shock half-life. |
| **EGARCH(1,1)** | Econometric | Variance | `p=1, q=1, dist='normal'` | Asymmetric leverage without non-negative constraints. |
| **GJR-GARCH(1,1)** | Econometric | Variance | `p=1, o=1, q=1, dist='normal'` | Threshold asymmetric response to negative shocks. |
| **HAR-RV (Corsi)** | Econometric | Realized Var | `daily=1, weekly=5, monthly=22` | Multi-component cascade mimicking long memory. |

---

## 2. Machine Learning Models

All machine learning models are trained using strictly causal feature matrices with standardization fitted on training sets only.

| Model | Family | Regularization / Constraints | Key Strengths |
|---|---|---|---|
| **Ridge Regression** | Regularized Linear | $L_2$ penalty (`alpha=1.0`) | Handles multicollinearity among technical indicators. |
| **Lasso Regression** | Regularized Linear | $L_1$ penalty (`alpha=0.01`) | Automatic feature selection and sparse representations. |
| **ElasticNet** | Regularized Linear | $L_1 + L_2$ penalty | Combines grouping effect with sparsity. |
| **Random Forest** | Bagging Ensemble | `n_estimators=150, max_depth=6` | Non-linear feature capture without overfitting. |
| **Extra Trees** | Randomized Trees | `n_estimators=150, max_depth=6` | Reduced variance and faster training. |
| **Gradient Boosting** | Boosting Ensemble | `n_estimators=120, lr=0.03, depth=4`| Sequential gradient descent optimization. |
| **LightGBM** | Leaf-wise GBDT | `n_estimators=120, lr=0.03, depth=4`| High throughput and memory efficiency. |
| **XGBoost** | Exact GBDT | `n_estimators=120, lr=0.03, depth=4`| Second-order Taylor loss approximation. |
| **Support Vector Reg.** | Kernel Methods | `kernel='rbf', C=1.0, epsilon=0.05` | High-dimensional non-linear mapping. |
| **Quantile GBDT** | Uncertainty Reg. | `alpha in [0.10, 0.50, 0.90]` | Predictive confidence intervals. |

---

## 3. Deep Learning Sequential Architectures

| Architecture | Sequential Mechanism | Receptive Field | Design Focus |
|---|---|---|---|
| **LSTM** | Gated memory cells | Long sequential memory | Mitigates vanishing gradient in long historical paths. |
| **GRU** | Reset & Update gates | Moderately long memory | Faster convergence and fewer parameters than LSTM. |
| **TCN** | Dilated causal convolutions | Exponential receptive field | Strict no-leakage causal filtering with parallel convolutions. |
| **Attention** | Multi-Head Self-Attention | Global pairwise correlations | Discovers latent dependencies across past volatility regimes. |
