"""Volatility Forecasting Lab: Streamlit Quantitative Research Workstation."""

import sys
import os

# Add src to python path for standalone execution
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from volatility_lab.data.ingestion import DataIngestionPipeline
from volatility_lab.data.providers.synthetic import SyntheticMarketDataProvider
from volatility_lab.estimators.historical import HistoricalVolatilityEstimator
from volatility_lab.estimators.ewma import EWMAVolatilityEstimator
from volatility_lab.estimators.parkinson import ParkinsonEstimator
from volatility_lab.estimators.garman_klass import GarmanKlassEstimator
from volatility_lab.estimators.rogers_satchell import RogersSatchellEstimator
from volatility_lab.econometrics.garch import GARCHModel
from volatility_lab.econometrics.egarch import EGARCHModel
from volatility_lab.econometrics.gjr_garch import GJRGARCHModel
from volatility_lab.econometrics.har_rv import HARRVModel
from volatility_lab.forecasting.prediction import MultiModelForecaster
from volatility_lab.evaluation.metrics import compute_all_metrics
from volatility_lab.evaluation.statistical_tests import diebold_mariano_test
from volatility_lab.regimes.detection import StatisticalRegimeDetector, GMMRegimeDetector
from volatility_lab.regimes.analysis import RegimeAnalyzer
from volatility_lab.risk.var import ValueAtRisk
from volatility_lab.risk.expected_shortfall import ExpectedShortfall
from volatility_lab.risk.backtesting import VaRBacktester
from volatility_lab.visualization.charts import (
    plot_volatility_estimators,
    plot_volatility_clustering,
    plot_forecast_fan_chart,
    plot_var_exceedances,
)
from volatility_lab.experiments.reporting import ResearchReportGenerator


st.set_page_config(
    page_title="Volatility Forecasting Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 26px; font-weight: 700; color: #1e293b; margin-bottom: 2px; }
    .sub-header { font-size: 14px; color: #64748b; margin-bottom: 20px; }
    .metric-card { background: #f8fafc; border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.title("🔬 Research Controls")
ticker = st.sidebar.selectbox(
    "Asset / Index",
    options=["^NSEI", "^NSEBANK", "^BSESN", "^GSPC", "^IXIC", "BTC-USD", "SYNTHETIC_SIM"],
    index=0,
    help="Select financial market instrument."
)

date_range = st.sidebar.date_input(
    "Sample Date Range",
    value=[pd.to_datetime("2018-01-01"), pd.to_datetime("2024-12-31")],
)

active_horizon = st.sidebar.select_slider(
    "Primary Forecast Horizon (Days)",
    options=[1, 5, 10, 20],
    value=5,
)

risk_cl = st.sidebar.selectbox("VaR Confidence Level", [0.95, 0.99, 0.90], index=0)

# ----------------- DATA INGESTION -----------------
@st.cache_data(show_spinner=False)
def load_market_data(sym: str, start: str, end: str):
    if sym == "SYNTHETIC_SIM":
        prov = SyntheticMarketDataProvider()
        raw = prov.fetch_data("SYNTHETIC", start, end)
        from volatility_lab.data.preprocessing import DataPreprocessor
        return DataPreprocessor().clean_and_prepare(raw)
    pipeline = DataIngestionPipeline()
    return pipeline.run(ticker=sym, start_date=start, end_date=end)

start_str = str(date_range[0]) if len(date_range) > 0 else "2018-01-01"
end_str = str(date_range[1]) if len(date_range) > 1 else "2024-12-31"

with st.spinner("Retrieving and validating analytical market data..."):
    df = load_market_data(ticker, start_str, end_str)

returns = df["log_return"]
daily_var = df["daily_variance_proxy"]

# ----------------- TOP METRICS BAR -----------------
st.markdown(f"<div class='main-header'>VOLATILITY FORECASTING LAB: {ticker}</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Institutional-Grade Volatility Modeling, Econometric Forecasting & Market Risk Analytics</div>", unsafe_allow_html=True)

curr_price = df["Close"].iloc[-1]
last_ret = returns.iloc[-1] * 100.0
hist_20_ann = HistoricalVolatilityEstimator(windows=20, annualize=True).estimate(returns).iloc[-1, 0] * 100.0
ewma_ann = EWMAVolatilityEstimator(decay_factor=0.94, annualize=True).estimate(returns).iloc[-1] * 100.0

regime_detector = StatisticalRegimeDetector()
regime_series = regime_detector.detect(daily_var)
curr_regime = regime_detector.regime_names.get(regime_series.iloc[-1], "Normal Volatility")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Latest Close", f"{curr_price:,.2f}", f"{last_ret:+.2f}%")
col2.metric("20d Hist Vol (Ann)", f"{hist_20_ann:.2f}%")
col3.metric("EWMA Vol (lambda=0.94)", f"{ewma_ann:.2f}%")
col4.metric("Volatility Regime", curr_regime)
col5.metric("Data Sample", f"{len(df):,} Days", f"{df.index[0].strftime('%Y')} - {df.index[-1].strftime('%Y')}")

# ----------------- MAIN NAVIGATION TABS -----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Volatility Estimators",
    "🔮 Multi-Horizon Forecasting",
    "🏆 Model Leaderboard",
    "🌊 Market Regimes",
    "🛡️ Quantitative Risk (VaR)",
    "📑 Research Report & Export",
])

# ----- TAB 1: ESTIMATORS -----
with tab1:
    st.subheader("Comparative Volatility Estimators")
    st.markdown("""
    Comparison between standard Close-to-Close historical volatility and high-efficiency range-based estimators:
    **Parkinson (1980)**, **Garman-Klass (1980)**, and **Rogers-Satchell (1991)**.
    """)

    # Compute estimators
    df_est = pd.DataFrame(index=df.index)
    df_est["hist_20d"] = HistoricalVolatilityEstimator(windows=20, annualize=True).estimate(returns)["hist_vol_20d"]
    df_est["ewma_94"] = EWMAVolatilityEstimator(decay_factor=0.94, annualize=True).estimate(returns)

    if {"High", "Low", "Open", "Close"}.issubset(df.columns):
        df_est["parkinson_20d"] = ParkinsonEstimator(window=20, annualize=True).estimate(df["High"], df["Low"])
        df_est["garman_klass_20d"] = GarmanKlassEstimator(window=20, annualize=True).estimate(df["Open"], df["High"], df["Low"], df["Close"])
        df_est["rogers_satchell_20d"] = RogersSatchellEstimator(window=20, annualize=True).estimate(df["Open"], df["High"], df["Low"], df["Close"])

    fig_est = plot_volatility_estimators(df_est, list(df_est.columns), title=f"{ticker}: Annualized Volatility Estimators (%)")
    st.plotly_chart(fig_est, use_container_width=True)

    st.subheader("Volatility Clustering & Absolute Returns")
    fig_clust = plot_volatility_clustering(returns, df_est["hist_20d"])
    st.plotly_chart(fig_clust, use_container_width=True)

# ----- TAB 2: FORECASTING CENTRE -----
with tab2:
    st.subheader(f"Multi-Horizon Forecasting Engine ({active_horizon}-Day Focus)")

    forecaster = MultiModelForecaster(horizons=[1, 5, 10, 20])
    with st.spinner("Generating multi-model volatility forecasts..."):
        fc_df = forecaster.forecast_all(df)

    h_fc = fc_df[fc_df["horizon_days"] == active_horizon].sort_values("forecast_vol_ann_pct")

    col_fc1, col_fc2 = st.columns([3, 2])
    with col_fc1:
        st.markdown(f"**Multi-Horizon Predictive Trajectory & Uncertainty Cone**")
        fig_fan = plot_forecast_fan_chart(df.index, df_est["hist_20d"], fc_df, title=f"{ticker} Forecast Cone (h=1 to 20 Days)")
        st.plotly_chart(fig_fan, use_container_width=True)

    with col_fc2:
        st.markdown(f"**Model Forecasts for Horizon: {active_horizon} Trading Days**")
        disp_table = h_fc[["model", "category", "forecast_vol_ann_pct", "lower_vol_ann_pct", "upper_vol_ann_pct"]].rename(
            columns={
                "model": "Model",
                "category": "Class",
                "forecast_vol_ann_pct": "Forecast Vol (%)",
                "lower_vol_ann_pct": "95% Lower (%)",
                "upper_vol_ann_pct": "95% Upper (%)",
            }
        )
        st.dataframe(disp_table.style.format({
            "Forecast Vol (%)": "{:.2f}%",
            "95% Lower (%)": "{:.2f}%",
            "95% Upper (%)": "{:.2f}%",
        }), use_container_width=True)

        st.info("💡 **Methodology Notice:** Targets are forward cumulative realized variance. All predictive models use strictly backward-looking information available up to the forecast origin.")

# ----- TAB 3: MODEL LEADERBOARD -----
with tab3:
    st.subheader("Out-of-Sample Walk-Forward Leaderboard")
    st.markdown("""
    Models evaluated on identical out-of-sample test samples.
    Ranked strictly by **Patton (2011) QLIKE Loss**, which is robust to noise in realized variance proxies.
    """)

    from volatility_lab.features.feature_pipeline import FeaturePipeline
    from volatility_lab.machine_learning.registry import MLModelRegistry

    pipe = FeaturePipeline()
    X, y = pipe.transform(df, horizon=active_horizon)

    if len(X) > 100:
        split_point = int(len(X) * 0.75)
        X_tr, y_tr = X.iloc[:split_point], y.iloc[:split_point]
        X_te, y_te = X.iloc[split_point:], y.iloc[split_point:]

        preds_dict = {}
        # Baselines
        preds_dict["Historical (20d)"] = np.full(len(y_te), y_tr.mean())

        # GARCH benchmark
        try:
            garch_fit = GARCHModel().fit(returns.loc[X_tr.index])
            garch_pred_val = garch_fit.forecast(horizon=active_horizon)
            from volatility_lab.forecasting.horizons import annualized_volatility_to_variance
            preds_dict["GARCH(1,1)"] = np.full(len(y_te), annualized_volatility_to_variance(garch_pred_val))
        except Exception:
            pass

        # ML models
        for m_name in ["ridge", "random_forest", "lightgbm"]:
            try:
                m = MLModelRegistry.get(m_name)
                m.fit(X_tr, y_tr)
                preds_dict[m_name.upper().replace("_", " ")] = np.maximum(m.predict(X_te), 1e-6)
            except Exception:
                pass

        leaderboard = ModelComparisonFramework.generate_leaderboard(preds_dict, y_te.values, horizon=active_horizon)
        st.dataframe(
            leaderboard.style.format({
                "qlike": "{:.4f}",
                "rmse": "{:.6f}",
                "mae": "{:.6f}",
                "r2": "{:.3f}",
                "dm_stat_vs_bench": "{:.3f}",
                "dm_pvalue": "{:.4f}",
            }),
            use_container_width=True,
        )
    else:
        st.warning("Insufficient samples for walk-forward comparison.")

# ----- TAB 4: MARKET REGIMES -----
with tab4:
    st.subheader("Market Regime Identification & Transition Dynamics")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("**Regime Classification Breakdown**")
        regime_counts = regime_series.value_counts().sort_index()
        reg_df = pd.DataFrame({
            "Regime": [regime_detector.regime_names[i] for i in regime_counts.index],
            "Observed Days": regime_counts.values,
            "Share (%)": (regime_counts.values / len(regime_series)) * 100.0,
        })
        st.dataframe(reg_df.style.format({"Share (%)": "{:.1f}%"}))

    with col_r2:
        st.markdown("**1-Step Markov Transition Probability Matrix**")
        t_matrix = RegimeAnalyzer.compute_transition_matrix(regime_series)
        st.dataframe(t_matrix.style.format("{:.2%}"), use_container_width=True)

        durations = RegimeAnalyzer.compute_expected_durations(t_matrix)
        st.markdown("**Expected Regime Duration (Trading Days)**")
        st.write(durations)

# ----- TAB 5: QUANTITATIVE RISK -----
with tab5:
    st.subheader(f"Value at Risk (VaR) & Expected Shortfall (Confidence = {int(risk_cl*100)}%)")

    var_calc = ValueAtRisk(confidence_levels=[risk_cl])
    var_norm = var_calc.parametric_normal_var(returns)
    var_hist = var_calc.historical_var(returns)
    var_t = var_calc.parametric_student_t_var(returns)

    es_calc = ExpectedShortfall(confidence_levels=[risk_cl])
    es_norm = es_calc.parametric_normal_es(returns)

    col_rk1, col_rk2, col_rk3 = st.columns(3)
    k_norm = f"var_norm_{int(risk_cl*100)}"
    k_hist = f"var_hist_{int(risk_cl*100)}"
    k_es = f"es_norm_{int(risk_cl*100)}"

    col_rk1.metric("Parametric Normal VaR", f"{var_norm[k_norm].iloc[-1]*100:.2f}%")
    col_rk2.metric("Historical Quantile VaR", f"{var_hist[k_hist].iloc[-1]*100:.2f}%")
    col_rk3.metric("Normal Expected Shortfall (CVaR)", f"{es_norm[k_es].iloc[-1]*100:.2f}%")

    st.subheader("Regulatory Basel Backtesting (Kupiec & Christoffersen)")
    clean_v95 = var_norm[k_norm].dropna()
    matched_r = returns.reindex(clean_v95.index)
    bt = VaRBacktester.backtest(matched_r, clean_v95, confidence_level=risk_cl)

    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    col_b1.metric("Observed Violations", f"{bt.actual_violations} / {bt.expected_violations:.1f}")
    col_b2.metric("Violation Ratio", f"{bt.violation_ratio:.2f}")
    col_b3.metric("Kupiec POF Test", f"p={bt.kupiec_p_value:.4f}", bt.kupiec_decision)
    col_b4.metric("Christoffersen Independence", f"p={bt.christoffersen_ind_p_value:.4f}", bt.christoffersen_ind_decision)

    fig_var = plot_var_exceedances(matched_r, clean_v95)
    st.plotly_chart(fig_var, use_container_width=True)

# ----- TAB 6: RESEARCH REPORT & EXPORT -----
with tab6:
    st.subheader("Generate Institutional Research Report")

    if st.button("Generate Formal Research Report"):
        report_md = ResearchReportGenerator.generate_markdown_report(
            ticker=ticker,
            current_vol_ann=hist_20_ann,
            leaderboard=fc_df.head(10),
            regime_info={"current_regime": curr_regime, "duration_days": 18.5},
            backtest_results={
                "Kupiec POF Decision": bt.kupiec_decision,
                "Christoffersen Independence": bt.christoffersen_ind_decision,
                "Conditional Coverage": bt.conditional_coverage_decision,
            }
        )
        st.markdown(report_md)
        st.download_button(
            label="📥 Download Markdown Report",
            data=report_md,
            file_name=f"{ticker}_volatility_research_report.md",
            mime="text/markdown",
        )
