"""Composite institutional trading decision and risk signal engine."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class InstitutionalSignalEngine:
    """Combines econometric volatility forecasts, microstructure range estimators,

    Barndorff-Nielsen jump dynamics, and VaR risk limits to generate institutional
    decision confirmations (Dark Green BUY / Dark Red SELL).
    """

    @staticmethod
    def evaluate(
        current_row: pd.Series,
        prev_row: Optional[pd.Series] = None,
        history_df: Optional[pd.DataFrame] = None,
        target_annualized_vol: float = 15.0,
    ) -> Dict[str, Any]:
        """Evaluates composite institutional signals from real-time or historical bar metrics.

        Args:
            current_row: Latest series observation with prices and volatility estimators.
            prev_row: Preceding bar observation for differential rate-of-change analysis.
            history_df: Optional trailing window DataFrame for percentile rank evaluation.
            target_annualized_vol: Target volatility (%) for volatility-targeted position sizing.

        Returns:
            Dictionary with signal verdict, color coding, confidence score, dynamic stop-loss,
            recommended position sizing, and checklist of verified conditions.
        """
        if prev_row is None:
            prev_row = current_row

        price = float(current_row.get("Close", 100.0))
        ret = float(current_row.get("log_return", 0.0))
        ewma_vol = float(current_row.get("ewma_vol_ann", 18.0))
        prev_ewma = float(prev_row.get("ewma_vol_ann", ewma_vol))
        rolling_rv = float(current_row.get("rolling_rv_ann", ewma_vol))
        gk_vol = float(current_row.get("garman_klass_ann", ewma_vol))
        prev_gk = float(prev_row.get("garman_klass_ann", gk_vol))
        jump_comp = float(current_row.get("jump_component", 0.0))
        var_95 = float(current_row.get("var_95_pct", 2.0))
        regime = int(current_row.get("volatility_regime", 1))

        # Check for price relative to moving average if present
        sma_val = float(current_row.get("sma_9", price)) if "sma_9" in current_row else price
        is_above_sma = price >= sma_val

        # Scaling factor for bar-level VaR
        bar_scale = np.sqrt(375.0) if ewma_vol > 0 else 1.0
        bar_var_95 = var_95 / bar_scale

        # Condition checks
        c_trend_up = ret > -0.0005 and is_above_sma
        c_trend_down = ret < 0.0 and price < sma_val
        c_vol_stable = regime in [0, 1]  # Low or Normal volatility
        c_vol_spike = regime in [2, 3]  # Elevated or Extreme volatility
        c_range_expanding = gk_vol > (1.02 * prev_gk)
        c_no_jump = jump_comp < 0.20
        c_jump_shock = jump_comp >= 0.25
        c_var_breach = (ret * 100.0) < -bar_var_95

        # Dynamic Stop Loss & Target calculation
        # Stop loss based on 95% Value at Risk threshold
        stop_loss_pts = price * (bar_var_95 / 100.0) * 1.25
        stop_loss_price = max(price - stop_loss_pts, 0.01)
        take_profit_price = price + (stop_loss_pts * 2.0)  # 2:1 Reward to Risk

        # Volatility-targeted position weight (e.g. 15% target / current vol)
        vol_weight = min(1.0, max(0.1, target_annualized_vol / max(ewma_vol, 1.0)))

        # ----------------- DECISION MATRIX EVALUATION -----------------
        # 1. DARK RED: CRITICAL SELL / ASYMMETRIC CRASH / RISK-OFF EXIT
        if c_var_breach or (c_trend_down and (c_jump_shock or c_vol_spike)):
            score = 85.0 if (c_var_breach and c_jump_shock) else 75.0
            return {
                "signal_type": "SELL",
                "badge": "CRITICAL SELL / RISK-OFF EXIT",
                "color": "dark_red",
                "bg_gradient": "linear-gradient(135deg, #450a0a 0%, #7f1d1d 50%, #991b1b 100%)",
                "border_color": "#ef4444",
                "text_color": "#fef2f2",
                "status_icon": "🚨",
                "confidence_score": score,
                "summary": "Downside Value at Risk breach or asymmetric volatility shock detected. Liquidate long exposure or deploy downside protective hedges.",
                "action": "LIQUIDATE LONGS / DEPLOY INTRADAY PUT HEDGES",
                "stop_loss": round(price + stop_loss_pts, 2),
                "target_price": round(price - (stop_loss_pts * 2.0), 2),
                "vol_weight": 0.0,
                "checklist": [
                    {"label": "Intraday 95% VaR Boundary", "passed": not c_var_breach, "detail": f"Return {ret*100:+.2f}% vs Limit -{bar_var_95:.2f}%"},
                    {"label": "Jump Shock Immunity", "passed": not c_jump_shock, "detail": f"Discrete Jump {jump_comp*100:.1f}% (Threshold 25%)"},
                    {"label": "Volatility Regime Stability", "passed": not c_vol_spike, "detail": f"Regime Level: {regime} (0=Low, 3=Extreme)"},
                    {"label": "Trend Direction", "passed": c_trend_up, "detail": f"Price {price:,.2f} vs SMA {sma_val:,.2f}"},
                ],
            }

        # 2. DARK GREEN: CONFIRMED BUY / VOLATILITY EXPANSION BREAKOUT
        elif c_trend_up and c_vol_stable and (c_range_expanding or ret > 0.0) and c_no_jump:
            score = 90.0 if (c_range_expanding and ret > 0.001) else 80.0
            return {
                "signal_type": "BUY",
                "badge": "CONFIRMED BUY / LONG ENTRY",
                "color": "dark_green",
                "bg_gradient": "linear-gradient(135deg, #022c22 0%, #064e3b 50%, #065f46 100%)",
                "border_color": "#10b981",
                "text_color": "#ecfdf5",
                "status_icon": "🟢",
                "confidence_score": score,
                "summary": "Positive momentum breakout validated by range expansion in stable low/normal volatility regime without discrete jump shocks.",
                "action": "INITIATE LONG POSITION (VOLATILITY TARGETED)",
                "stop_loss": round(stop_loss_price, 2),
                "target_price": round(take_profit_price, 2),
                "vol_weight": round(vol_weight * 100.0, 1),
                "checklist": [
                    {"label": "Trend Alignment (Price > SMA)", "passed": True, "detail": f"Price {price:,.2f} >= SMA {sma_val:,.2f}"},
                    {"label": "Volatility Regime Stability", "passed": True, "detail": f"Regime Level: {regime} (Non-Crash Environment)"},
                    {"label": "Range Expansion Confirmation", "passed": c_range_expanding, "detail": f"Garman-Klass Vol: {gk_vol:.2f}% (vs Prior {prev_gk:.2f}%)"},
                    {"label": "Continuous Flow (No Jump Shock)", "passed": True, "detail": f"Jump Component: {jump_comp*100:.1f}% (Safe < 20%)"},
                ],
            }

        # 3. DARK EMERALD / GREEN: VOLATILITY SPIKE EXHAUSTION (REBOUND ACCUMULATION)
        elif regime == 3 and ewma_vol < prev_ewma and c_no_jump:
            return {
                "signal_type": "BUY_REVERSAL",
                "badge": "CAPITULATION REVERSAL / ACCUMULATE",
                "color": "dark_green",
                "bg_gradient": "linear-gradient(135deg, #064e3b 0%, #047857 50%, #059669 100%)",
                "border_color": "#34d399",
                "text_color": "#ecfdf5",
                "status_icon": "⚡",
                "confidence_score": 75.0,
                "summary": "Extreme volatility spike peaked and is mean-reverting downward. Discrete jump shocks have dissipated, signaling statistical capitulation bottom.",
                "action": "SCALE INTO MEAN-REVERSION REVERSAL LONGS / SELL OVERPRICED VOLATILITY",
                "stop_loss": round(stop_loss_price, 2),
                "target_price": round(price + (stop_loss_pts * 1.5), 2),
                "vol_weight": round(vol_weight * 60.0, 1),  # Half size on mean-reversion
                "checklist": [
                    {"label": "Volatility Peak Reversal", "passed": True, "detail": f"EWMA Vol falling ({ewma_vol:.2f}% < {prev_ewma:.2f}%)"},
                    {"label": "Extreme Regime Exhaustion", "passed": True, "detail": "Regime 3 reached; mean-reversion active"},
                    {"label": "Shock Dissipation", "passed": True, "detail": f"Jump ratio subsided to {jump_comp*100:.1f}%"},
                    {"label": "Risk Asymmetry", "passed": True, "detail": "High risk-reward skew on volatility compression"},
                ],
            }

        # 4. NEUTRAL: CONSOLIDATION / MONITORING
        else:
            return {
                "signal_type": "NEUTRAL",
                "badge": "NEUTRAL / MONITOR CONSOLIDATION",
                "color": "dark_slate",
                "bg_gradient": "linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%)",
                "border_color": "#64748b",
                "text_color": "#f8fafc",
                "status_icon": "⚪",
                "confidence_score": 50.0,
                "summary": "Market in range-bound consolidation with balanced risk-reward. Volatility compression and directional momentum currently neutral.",
                "action": "MAINTAIN DISCIPLINED HOLD / WAIT FOR RANGE BREAKOUT OR SPIKE",
                "stop_loss": round(stop_loss_price, 2),
                "target_price": round(take_profit_price, 2),
                "vol_weight": round(vol_weight * 50.0, 1),
                "checklist": [
                    {"label": "Trend Definition", "passed": c_trend_up, "detail": f"Price {price:,.2f} vs SMA {sma_val:,.2f}"},
                    {"label": "Volatility Regime", "passed": c_vol_stable, "detail": f"Current Regime: {regime}"},
                    {"label": "Range Expansion Trigger", "passed": c_range_expanding, "detail": f"GK {gk_vol:.2f}% vs Prev {prev_gk:.2f}%"},
                    {"label": "Risk Envelope Status", "passed": not c_var_breach, "detail": f"Return {ret*100:+.2f}% within VaR bounds"},
                ],
            }
