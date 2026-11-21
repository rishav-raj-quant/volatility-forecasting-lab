"""Market data integrity validation module."""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from volatility_lab.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """Contains statistical and structural validation outcomes for market data."""
    is_valid: bool
    total_records: int
    missing_values: Dict[str, int]
    zero_or_negative_prices: int
    invalid_high_low: int
    invalid_open_close: int
    duplicate_timestamps: int
    non_monotonic_timestamps: bool
    outlier_count: int
    issues: List[str] = field(default_factory=list)


class DataValidator:
    """Enforces strict financial time-series integrity constraints."""

    def __init__(self, outlier_return_threshold: float = 0.25):
        self.outlier_threshold = outlier_return_threshold

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        issues = []
        if df.empty:
            return ValidationReport(
                is_valid=False,
                total_records=0,
                missing_values={},
                zero_or_negative_prices=0,
                invalid_high_low=0,
                invalid_open_close=0,
                duplicate_timestamps=0,
                non_monotonic_timestamps=False,
                outlier_count=0,
                issues=["DataFrame is empty."],
            )

        # Datetime index validation
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append("DataFrame index must be a DatetimeIndex.")

        duplicate_ts = int(df.index.duplicated().sum())
        if duplicate_ts > 0:
            issues.append(f"Found {duplicate_ts} duplicate timestamps.")

        is_monotonic = df.index.is_monotonic_increasing
        if not is_monotonic:
            issues.append("Timestamp index is not strictly monotonically increasing.")

        # Missing values
        missing = {col: int(df[col].isna().sum()) for col in df.columns}
        total_missing = sum(missing.values())
        if total_missing > 0:
            issues.append(f"Total {total_missing} missing cells detected across columns.")

        # Price feasibility checks
        price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
        invalid_prices = 0
        for col in price_cols:
            invalid_prices += int((df[col] <= 0).sum())
        if invalid_prices > 0:
            issues.append(f"Detected {invalid_prices} non-positive price observations.")

        # High / Low boundary bounds
        invalid_hl = 0
        invalid_oc = 0
        if {"High", "Low"}.issubset(df.columns):
            invalid_hl = int((df["High"] < df["Low"]).sum())
            if invalid_hl > 0:
                issues.append(f"Detected {invalid_hl} rows where High < Low.")

        if {"High", "Open", "Close", "Low"}.issubset(df.columns):
            h_viol = (df["High"] < df["Open"]) | (df["High"] < df["Close"])
            l_viol = (df["Low"] > df["Open"]) | (df["Low"] > df["Close"])
            invalid_oc = int((h_viol | l_viol).sum())
            if invalid_oc > 0:
                issues.append(f"Detected {invalid_oc} rows violating High >= max(O,C) or Low <= min(O,C).")

        # Outlier diagnostics (unadjusted single-day extreme moves)
        outlier_count = 0
        if "Close" in df.columns and len(df) > 1:
            log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
            outliers = (np.abs(log_ret) > self.outlier_threshold)
            outlier_count = int(outliers.sum())
            if outlier_count > 0:
                issues.append(f"Flagged {outlier_count} single-day log return moves exceeding {self.outlier_threshold * 100:.0f}%.")

        is_valid = len(issues) == 0 or (outlier_count > 0 and len(issues) == 1)

        return ValidationReport(
            is_valid=is_valid,
            total_records=len(df),
            missing_values=missing,
            zero_or_negative_prices=invalid_prices,
            invalid_high_low=invalid_hl,
            invalid_open_close=invalid_oc,
            duplicate_timestamps=duplicate_ts,
            non_monotonic_timestamps=not is_monotonic,
            outlier_count=outlier_count,
            issues=issues,
        )
