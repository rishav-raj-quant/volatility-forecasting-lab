"""Real-time market data acquisition, intraday streaming, and continuous volatility processing."""

from typing import Dict, List, Optional, Tuple, Any
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

from volatility_lab.utils.logging import get_logger
from volatility_lab.utils.calendar import SQRT_252

logger = get_logger(__name__)


class RealTimeVolatilityEngine:
    """Acquires live intraday market data and calculates real-time volatility metrics."""

    def __init__(self, annualize: bool = True):
        self.annualize = annualize

    def fetch_live_intraday(
        self,
        ticker: str,
        interval: str = "1m",
        period: str = "1d",
    ) -> pd.DataFrame:
        """Fetches live intraday OHLCV bars from market feeds.

        Args:
            ticker: Financial instrument symbol (e.g. 'BTC-USD', '^NSEI', '^GSPC').
            interval: Intraday frequency ('1m', '2m', '5m', '15m', '60m').
            period: Time window ('1d', '5d', '7d').

        Returns:
            Validated DataFrame with real-time volatility metrics.
        """
        logger.info(f"Fetching real-time intraday data for {ticker} [interval={interval}, period={period}]")
        try:
            df = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
            )
            if df.empty:
                # If 1d is empty (e.g. weekend/pre-market for equities), fetch 5d to get last active session
                logger.warning(f"1d empty for {ticker}, falling back to 5d period.")
                df = yf.download(
                    tickers=ticker,
                    period="5d",
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                )

            if df.empty:
                raise ValueError(f"No live intraday data available for ticker '{ticker}'.")

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index = pd.to_datetime(df.index)
            df.index.name = "Datetime"

            req_cols = ["Open", "High", "Low", "Close", "Volume"]
            for c in req_cols:
                if c not in df.columns:
                    raise KeyError(f"Missing required price column: {c}")

            clean_df = df[req_cols].dropna().astype(float)
            if len(clean_df) < 5:
                raise ValueError(f"Insufficient intraday observations ({len(clean_df)}) for {ticker}.")

            # Enrich with real-time volatility metrics
            return self.process_intraday_volatility(clean_df, ticker=ticker, interval=interval)

        except Exception as e:
            logger.error(f"Error fetching live intraday data for {ticker}: {e}")
            raise

    def process_intraday_volatility(
        self,
        df: pd.DataFrame,
        ticker: str = "UNKNOWN",
        interval: str = "1m",
        window: int = 20,
        decay_factor: float = 0.94,
    ) -> pd.DataFrame:
        """Enriches intraday OHLCV bars with rolling and instantaneous volatility metrics.

        Args:
            df: OHLCV DataFrame with DatetimeIndex.
            ticker: Ticker name.
            interval: Frequency ('1m', '5m', etc.).
            window: Rolling bar window for range estimators.
            decay_factor: EWMA decay factor lambda.
        """
        res = df.copy()

        # Log returns
        res["log_return"] = np.log(res["Close"] / res["Close"].shift(1)).fillna(0.0)

        # Scale factor: annualization
        # Bars per trading day:
        # 1m crypto: 1440 bars/day; 1m equity: 375 bars/day (09:15-15:30 IST or 09:30-16:00 US)
        # 5m equity: 75 bars/day; 15m: 25 bars/day
        bars_per_day = 1440 if "BTC" in ticker or "ETH" in ticker or "USD" in ticker else 375
        if interval == "5m":
            bars_per_day = bars_per_day // 5
        elif interval == "15m":
            bars_per_day = bars_per_day // 15
        elif interval == "60m" or interval == "1h":
            bars_per_day = bars_per_day // 60

        ann_factor = np.sqrt(252 * bars_per_day) if self.annualize else 1.0

        # 1. Rolling Realized Volatility (annualized %)
        rolling_var = (res["log_return"] ** 2).rolling(window=window, min_periods=3).mean()
        res["rolling_rv_ann"] = np.sqrt(rolling_var) * ann_factor * 100.0

        # 2. Cumulative Realized Volatility for current session
        res["cum_rv_ann"] = np.sqrt((res["log_return"] ** 2).expanding().mean()) * ann_factor * 100.0

        # 3. Real-Time EWMA Volatility
        returns_sq = (res["log_return"] ** 2).values
        n = len(returns_sq)
        ewma_var = np.zeros(n)
        initial_var = np.var(res["log_return"].iloc[:min(window, n)]) if n > 1 else 1e-4
        ewma_var[0] = initial_var if initial_var > 0 else 1e-4

        for t in range(1, n):
            ewma_var[t] = decay_factor * ewma_var[t - 1] + (1.0 - decay_factor) * returns_sq[t - 1]

        res["ewma_vol_ann"] = np.sqrt(ewma_var) * ann_factor * 100.0

        # 4. Parkinson Range Estimator
        hl_ratio = np.log(res["High"] / res["Low"]) ** 2
        parkinson_var = hl_ratio.rolling(window=window, min_periods=3).mean() / (4.0 * np.log(2.0))
        res["parkinson_ann"] = np.sqrt(parkinson_var) * ann_factor * 100.0

        # 5. Garman-Klass Range Estimator
        co_ratio = np.log(res["Close"] / res["Open"]) ** 2
        gk_comp = 0.5 * hl_ratio - (2.0 * np.log(2.0) - 1.0) * co_ratio
        gk_var = gk_comp.rolling(window=window, min_periods=3).mean().clip(lower=0.0)
        res["garman_klass_ann"] = np.sqrt(gk_var) * ann_factor * 100.0

        # 6. Rogers-Satchell Range Estimator (handles non-zero drift)
        ho = np.log(res["High"] / res["Open"])
        hc = np.log(res["High"] / res["Close"])
        lo = np.log(res["Low"] / res["Open"])
        lc = np.log(res["Low"] / res["Close"])
        rs_comp = (ho * hc) + (lo * lc)
        rs_var = rs_comp.rolling(window=window, min_periods=3).mean().clip(lower=0.0)
        res["rogers_satchell_ann"] = np.sqrt(rs_var) * ann_factor * 100.0

        # 7. Bipower Variation (BV) and Jump Detection
        abs_ret = np.abs(res["log_return"])
        bv_comp = (np.pi / 2.0) * (abs_ret * abs_ret.shift(1))
        rolling_bv = bv_comp.rolling(window=window, min_periods=3).mean()
        res["bipower_var_ann"] = np.sqrt(rolling_bv) * ann_factor * 100.0

        # Jump ratio: (RV^2 - BV) / RV^2
        rv_sq = (rolling_var * ann_factor ** 2).clip(lower=1e-8)
        bv_sq = (rolling_bv * ann_factor ** 2).clip(lower=1e-8)
        jump_ratio = np.maximum(0.0, (rv_sq - bv_sq) / rv_sq)
        res["jump_component"] = jump_ratio

        # 8. Real-Time Intraday Risk: 95% and 99% VaR & CVaR
        # Instantaneous daily-equivalent sigma
        instant_daily_sigma = np.sqrt(ewma_var) * np.sqrt(bars_per_day)
        res["var_95_pct"] = 1.645 * instant_daily_sigma * 100.0
        res["var_99_pct"] = 2.326 * instant_daily_sigma * 100.0
        res["cvar_95_pct"] = 2.062 * instant_daily_sigma * 100.0

        # Regime indicator: 0 = Low, 1 = Normal, 2 = Elevated, 3 = Extreme Spike
        p75 = np.percentile(res["ewma_vol_ann"].dropna(), 75) if len(res) > 10 else 25.0
        p95 = np.percentile(res["ewma_vol_ann"].dropna(), 95) if len(res) > 10 else 40.0
        regimes = np.where(
            res["ewma_vol_ann"] > p95,
            3,
            np.where(res["ewma_vol_ann"] > p75, 2, np.where(res["ewma_vol_ann"] > 15.0, 1, 0)),
        )
        res["volatility_regime"] = regimes

        return res.ffill().bfill()


class LiveStreamingSimulator:
    """Simulates a live high-frequency order/bar stream with dynamic volatility jumps and regime switching."""

    def __init__(
        self,
        ticker: str = "^NSEI",
        initial_price: float = 25000.0,
        baseline_vol_ann: float = 0.14,
        interval_minutes: int = 1,
    ):
        self.ticker = ticker
        self.current_price = initial_price
        self.baseline_vol = baseline_vol_ann
        self.interval_minutes = interval_minutes
        self.dt = interval_minutes / (252.0 * 375.0)  # Trading fraction
        self.variance = (baseline_vol_ann) ** 2

        # Stochastic Volatility (GJR-GARCH dynamics)
        self.omega = 0.05 * self.variance
        self.alpha = 0.08
        self.beta = 0.85
        self.gamma = 0.06  # Leverage asymmetry

        # Buffer of simulated bars
        self.history: List[Dict[str, Any]] = []
        self._init_history(n_bars=60)

    def _init_history(self, n_bars: int = 60) -> None:
        """Initializes a warm-up buffer of historical bars."""
        now = datetime.datetime.now() - datetime.timedelta(minutes=n_bars * self.interval_minutes)
        for i in range(n_bars):
            bar_time = now + datetime.timedelta(minutes=i * self.interval_minutes)
            self._generate_bar(bar_time, shock=0.0)

    def _generate_bar(self, timestamp: datetime.datetime, shock: float = 0.0) -> Dict[str, Any]:
        """Generates a realistic intraday OHLCV bar."""
        sigma = np.sqrt(max(self.variance, 1e-6))
        z = np.random.normal(0, 1)

        # Log return with optional discrete shock
        ret = (sigma * np.sqrt(self.dt) * z) + shock
        open_price = self.current_price
        close_price = open_price * np.exp(ret)

        # High & Low excursion via Brownian Bridge reflection
        high_excursion = abs(np.random.normal(0, 1)) * sigma * np.sqrt(self.dt) * 0.8
        low_excursion = abs(np.random.normal(0, 1)) * sigma * np.sqrt(self.dt) * 0.8

        high_price = max(open_price, close_price) * (1.0 + high_excursion)
        low_price = min(open_price, close_price) * (1.0 - low_excursion)

        # Volume proportional to price variance and shock
        base_vol = 50000
        vol_mult = 1.0 + 5.0 * abs(ret) + (10.0 if shock != 0.0 else 0.0)
        volume = int(base_vol * vol_mult * np.random.uniform(0.7, 1.4))

        # Update GJR-GARCH variance state
        indicator = 1.0 if ret < 0 else 0.0
        self.variance = (
            self.omega
            + self.alpha * (ret / np.sqrt(self.dt)) ** 2
            + self.gamma * (ret / np.sqrt(self.dt)) ** 2 * indicator
            + self.beta * self.variance
        )
        self.variance = np.clip(self.variance, 0.005 ** 2, 1.5 ** 2)
        self.current_price = close_price

        bar = {
            "Datetime": timestamp,
            "Open": open_price,
            "High": high_price,
            "Low": low_price,
            "Close": close_price,
            "Volume": volume,
        }
        self.history.append(bar)
        return bar

    def step(self, shock: float = 0.0) -> pd.DataFrame:
        """Appends the next real-time bar to the stream and returns enriched metrics.

        Args:
            shock: Optional sudden price shock (-0.02 for flash crash, +0.02 for rally).
        """
        now = datetime.datetime.now()
        self._generate_bar(now, shock=shock)
        df_raw = pd.DataFrame(self.history).set_index("Datetime")
        engine = RealTimeVolatilityEngine(annualize=True)
        return engine.process_intraday_volatility(
            df_raw,
            ticker=self.ticker,
            interval=f"{self.interval_minutes}m",
        )

    def get_current_dataframe(self) -> pd.DataFrame:
        """Returns the current simulated DataFrame enriched with metrics."""
        df_raw = pd.DataFrame(self.history).set_index("Datetime")
        engine = RealTimeVolatilityEngine(annualize=True)
        return engine.process_intraday_volatility(
            df_raw,
            ticker=self.ticker,
            interval=f"{self.interval_minutes}m",
        )


def compute_realtime_forecast_cones(
    current_vol_ann: float,
    horizon_bars: int = 30,
    long_run_vol_ann: float = 16.0,
    half_life_bars: int = 15,
) -> Dict[str, Any]:
    """Generates multi-step real-time forward volatility projections with confidence cones.

    Args:
        current_vol_ann: Instantaneous annualized volatility (%) at current bar.
        horizon_bars: Forward bars to project.
        long_run_vol_ann: Long-run unconditional volatility mean (%).
        half_life_bars: Mean reversion half-life in bars.

    Returns:
        Dictionary containing forecast path, upper/lower 68% and 95% fan bounds.
    """
    kappa = np.log(2.0) / max(half_life_bars, 1)
    bars = np.arange(1, horizon_bars + 1)

    theta = long_run_vol_ann ** 2
    curr_var = current_vol_ann ** 2

    # Mean reverting conditional variance trajectory: E[sigma_{t+k}^2] = theta + (sigma_t^2 - theta)*exp(-kappa*k)
    exp_factor = np.exp(-kappa * bars)
    pred_var = theta + (curr_var - theta) * exp_factor
    pred_vol = np.sqrt(np.maximum(pred_var, 1.0))

    # Uncertainty fan widening as sqrt(horizon)
    uncertainty_scale = 0.08 * pred_vol * np.sqrt(bars / 5.0)

    upper_95 = pred_vol + 1.96 * uncertainty_scale
    lower_95 = np.maximum(pred_vol - 1.96 * uncertainty_scale, 2.0)
    upper_68 = pred_vol + 1.00 * uncertainty_scale
    lower_68 = np.maximum(pred_vol - 1.00 * uncertainty_scale, 3.0)

    return {
        "bars": bars.tolist(),
        "mean_forecast": pred_vol.tolist(),
        "upper_95": upper_95.tolist(),
        "lower_95": lower_95.tolist(),
        "upper_68": upper_68.tolist(),
        "lower_68": lower_68.tolist(),
    }
