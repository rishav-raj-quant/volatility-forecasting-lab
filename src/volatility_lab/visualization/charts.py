"""Publication-grade interactive Plotly financial volatility visualizations."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_volatility_estimators(
    df: pd.DataFrame,
    vol_columns: List[str],
    title: str = "Volatility Estimator Comparison",
) -> go.Figure:
    """Plots comparative historical and range-based volatility series."""
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, col in enumerate(vol_columns):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col] * 100.0,
                    mode="lines",
                    name=col.replace("_", " ").title(),
                    line=dict(width=1.5, color=colors[i % len(colors)]),
                )
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Arial")),
        xaxis=dict(title="Date", showgrid=True, gridcolor="#e1e5ea"),
        yaxis=dict(title="Annualized Volatility (%)", showgrid=True, gridcolor="#e1e5ea"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=40, t=60, b=40),
        hovermode="x unified",
    )
    return fig


def plot_volatility_clustering(returns: pd.Series, rolling_vol: pd.Series) -> go.Figure:
    """Creates dual-pane plot showing return series and corresponding volatility clustering."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Daily Log Returns (r_t)", "20-Day Annualized Rolling Volatility (%)")
    )

    fig.add_trace(
        go.Scatter(x=returns.index, y=returns.values * 100.0, mode="lines", name="Log Return (%)", line=dict(color="#4682b4", width=1.0)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=rolling_vol.index, y=rolling_vol.values * 100.0, mode="lines", name="Rolling Vol (%)", line=dict(color="#d9534f", width=1.8)),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        margin=dict(l=50, r=40, t=50, b=40),
        hovermode="x unified",
    )
    return fig


def plot_forecast_fan_chart(
    historical_dates: pd.DatetimeIndex,
    historical_vol: pd.Series,
    forecast_df: pd.DataFrame,
    title: str = "Volatility Forecast Trajectory & 95% Predictive Cone",
) -> go.Figure:
    """Renders historical volatility tail and forward multi-horizon forecast cone."""
    fig = go.Figure()

    # Historical tail (last 60 trading days)
    tail_len = min(60, len(historical_dates))
    fig.add_trace(
        go.Scatter(
            x=historical_dates[-tail_len:],
            y=historical_vol.iloc[-tail_len:].values * 100.0,
            mode="lines",
            name="Observed Volatility",
            line=dict(color="#2b2d42", width=2.0),
        )
    )

    last_date = historical_dates[-1]
    # Filter for top model or specific model
    models = forecast_df["model"].unique()
    top_model = models[0] if len(models) > 0 else "Model"
    sub_df = forecast_df[forecast_df["model"] == top_model].sort_values("horizon_days")

    if not sub_df.empty:
        forward_days = [int(h) for h in sub_df["horizon_days"]]
        fc_dates = [last_date + pd.Timedelta(days=d) for d in forward_days]
        fc_vals = sub_df["forecast_vol_ann_pct"].values
        low_vals = sub_df["lower_vol_ann_pct"].values
        up_vals = sub_df["upper_vol_ann_pct"].values

        # Connect anchor to latest observed
        last_obs = historical_vol.iloc[-1] * 100.0
        cone_dates = [last_date] + fc_dates
        cone_fc = [last_obs] + list(fc_vals)
        cone_low = [last_obs] + list(low_vals)
        cone_up = [last_obs] + list(up_vals)

        # Upper bound
        fig.add_trace(
            go.Scatter(x=cone_dates, y=cone_up, mode="lines", line=dict(width=0), showlegend=False)
        )
        # Lower bound with fill
        fig.add_trace(
            go.Scatter(
                x=cone_dates,
                y=cone_low,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(30, 144, 255, 0.2)",
                line=dict(width=0),
                name="95% Predictive Interval",
            )
        )
        # Point Forecast
        fig.add_trace(
            go.Scatter(
                x=cone_dates,
                y=cone_fc,
                mode="lines+markers",
                name=f"Forecast ({top_model})",
                line=dict(color="#1e90ff", width=2.5, dash="dash"),
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(title="Timeline", showgrid=True),
        yaxis=dict(title="Annualized Volatility (%)", showgrid=True),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=40, t=60, b=40),
    )
    return fig


def plot_var_exceedances(
    returns: pd.Series,
    var_95: pd.Series,
    var_99: Optional[pd.Series] = None,
) -> go.Figure:
    """Plots daily losses with rolling VaR thresholds and highlights breaches."""
    losses = -returns
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=losses.index,
            y=losses.values * 100.0,
            mode="lines",
            name="Daily Loss (-r_t %)",
            line=dict(color="#adb5bd", width=1.0),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=var_95.index,
            y=var_95.values * 100.0,
            mode="lines",
            name="95% VaR Threshold",
            line=dict(color="#f39c12", width=1.8),
        )
    )

    if var_99 is not None:
        fig.add_trace(
            go.Scatter(
                x=var_99.index,
                y=var_99.values * 100.0,
                mode="lines",
                name="99% VaR Threshold",
                line=dict(color="#e74c3c", width=1.8),
            )
        )

    # Highlight 95% breaches
    breaches_95 = losses[losses > var_95]
    if not breaches_95.empty:
        fig.add_trace(
            go.Scatter(
                x=breaches_95.index,
                y=breaches_95.values * 100.0,
                mode="markers",
                name="VaR 95% Exceedance",
                marker=dict(color="#e74c3c", size=6, symbol="x"),
            )
        )

    fig.update_layout(
        title="Value at Risk (VaR) Exceedance Diagnostics",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Loss Percentage (%)"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=40, t=60, b=40),
    )
    return fig
