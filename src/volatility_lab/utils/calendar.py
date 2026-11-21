"""Market trading calendar utilities, annualization factors, and timezone conversions."""

from typing import Union
import pandas as pd
import pytz

ANNUALIZATION_FACTOR_DAILY = 252.0
SQRT_252 = 252.0 ** 0.5


def ensure_market_timezone(
    dt_index: pd.DatetimeIndex,
    target_tz: str = "Asia/Kolkata"
) -> pd.DatetimeIndex:
    """Ensures datetime index has appropriate market timezone localized or converted."""
    if dt_index.tz is None:
        return dt_index.tz_localize("UTC").tz_convert(target_tz)
    return dt_index.tz_convert(target_tz)


def get_annualization_factor(frequency: str = "daily") -> float:
    """Returns trading period multiplier for volatility annualization."""
    freq = frequency.lower()
    if freq in ("daily", "1d", "d"):
        return SQRT_252
    elif freq in ("hourly", "1h", "60m"):
        return (252.0 * 6.25) ** 0.5  # Standard NSE 6.25 hr session
    elif freq in ("5min", "5m"):
        return (252.0 * 75.0) ** 0.5
    elif freq in ("1min", "1m"):
        return (252.0 * 375.0) ** 0.5
    return SQRT_252
