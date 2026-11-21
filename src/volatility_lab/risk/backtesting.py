"""Regulatory VaR backtesting: Kupiec POF, Christoffersen Independence, and Conditional Coverage."""

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class VaRBacktestResult:
    """Comprehensive statistical backtest report for a VaR model."""
    confidence_level: float
    total_observations: int
    actual_violations: int
    expected_violations: float
    violation_ratio: float
    kupiec_lr_stat: float
    kupiec_p_value: float
    kupiec_decision: str
    christoffersen_ind_stat: float
    christoffersen_ind_p_value: float
    christoffersen_ind_decision: str
    conditional_coverage_stat: float
    conditional_coverage_p_value: float
    conditional_coverage_decision: str


class VaRBacktester:
    """Rigorous statistical validation of VaR forecasts according to Basel/BIS quantitative standards."""

    @staticmethod
    def backtest(
        returns: pd.Series,
        var_forecasts: pd.Series,
        confidence_level: float = 0.95,
    ) -> VaRBacktestResult:
        """Executes unconditional and conditional coverage backtests.

        A violation occurs when negative return (loss) exceeds VaR:
            I_t = 1 if -r_t > VaR_t else 0
        """
        combined = pd.DataFrame({"ret": returns, "var": var_forecasts}).dropna()
        n = len(combined)
        if n < 20:
            raise ValueError(f"Insufficient observations ({n}) for statistical VaR backtesting.")

        losses = -combined["ret"]
        hits = (losses > combined["var"]).astype(int).values
        x = int(hits.sum())
        p = 1.0 - confidence_level
        expected_x = n * p
        v_ratio = x / expected_x if expected_x > 0 else 0.0

        # 1. Kupiec Proportion of Failures (POF) Likelihood Ratio test
        # H0: p_hat = p
        p_hat = x / n
        if x == 0:
            lr_pof = -2.0 * np.log((1.0 - p) ** n)
        elif x == n:
            lr_pof = -2.0 * np.log(p ** n)
        else:
            log_num = (n - x) * np.log(1.0 - p) + x * np.log(p)
            log_den = (n - x) * np.log(1.0 - p_hat) + x * np.log(p_hat)
            lr_pof = -2.0 * (log_num - log_den)

        lr_pof = max(0.0, float(lr_pof))
        kupiec_pval = float(1.0 - stats.chi2.cdf(lr_pof, df=1))
        kupiec_dec = "Accept H0 (Accurate Coverage)" if kupiec_pval >= 0.05 else "Reject H0 (Inaccurate Coverage)"

        # 2. Christoffersen (1998) Independence Test
        # Contingency matrix of transitions
        n00, n01, n10, n11 = 0, 0, 0, 0
        for t in range(n - 1):
            if hits[t] == 0 and hits[t + 1] == 0:
                n00 += 1
            elif hits[t] == 0 and hits[t + 1] == 1:
                n01 += 1
            elif hits[t] == 1 and hits[t + 1] == 0:
                n10 += 1
            elif hits[t] == 1 and hits[t + 1] == 1:
                n11 += 1

        pi_0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
        pi_1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
        pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0

        # Unrestricted likelihood vs Restricted (independence) likelihood
        def safe_log(val):
            return np.log(val) if val > 0 else 0.0

        ll_unrestricted = (
            n00 * safe_log(1.0 - pi_0)
            + n01 * safe_log(pi_0)
            + n10 * safe_log(1.0 - pi_1)
            + n11 * safe_log(pi_1)
        )
        ll_restricted = (n00 + n10) * safe_log(1.0 - pi) + (n01 + n11) * safe_log(pi)

        lr_ind = -2.0 * (ll_restricted - ll_unrestricted)
        lr_ind = max(0.0, float(lr_ind))
        ind_pval = float(1.0 - stats.chi2.cdf(lr_ind, df=1))
        ind_dec = "Accept H0 (Independent Exceedances)" if ind_pval >= 0.05 else "Reject H0 (Clustered Exceedances)"

        # 3. Conditional Coverage Test: LR_cc = LR_pof + LR_ind ~ Chi2(2)
        lr_cc = lr_pof + lr_ind
        cc_pval = float(1.0 - stats.chi2.cdf(lr_cc, df=2))
        cc_dec = "Accept H0 (Valid Model)" if cc_pval >= 0.05 else "Reject H0 (Invalid Model)"

        return VaRBacktestResult(
            confidence_level=confidence_level,
            total_observations=n,
            actual_violations=x,
            expected_violations=expected_x,
            violation_ratio=float(v_ratio),
            kupiec_lr_stat=lr_pof,
            kupiec_p_value=kupiec_pval,
            kupiec_decision=kupiec_dec,
            christoffersen_ind_stat=lr_ind,
            christoffersen_ind_p_value=ind_pval,
            christoffersen_ind_decision=ind_dec,
            conditional_coverage_stat=lr_cc,
            conditional_coverage_p_value=cc_pval,
            conditional_coverage_decision=cc_dec,
        )
