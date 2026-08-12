"""
CryptoCast - Multi-Timeframe Hypothesis Strategy Engine
========================================================
Executes Welch's t-test and Mann-Whitney U tests on Daily and Weekly timeframes.
Applies statistical significance gates (p-value < alpha, t-stat > 0) to filter
trading simulator entries.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def run_weekly_hypothesis_test(df: pd.DataFrame) -> Tuple[float, float, float, float, int]:
    """Run Welch's t-test and Mann-Whitney U test for Weekly Open Discount Strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing ['Price', 'Open', 'Forward_7D_Return'].

    Returns
    -------
    Tuple[float, float, float, float, int]
        (t_stat, p_value_t, u_stat, p_value_u, n_samples)
    """
    df = df.copy()
    if "Week_Open" not in df.columns:
        week_opens = []
        for idx_dt in df.index:
            monday_dt = idx_dt - pd.Timedelta(days=idx_dt.dayofweek)
            match = df[df.index >= monday_dt]
            week_opens.append(
                match["Open"].iloc[0] if not match.empty else df.loc[idx_dt, "Open"]
            )
        df["Week_Open"] = week_opens

    if "Forward_7D_Return" not in df.columns:
        df["Forward_7D_Return"] = (
            (df["Price"].shift(-7) - df["Price"]) / df["Price"] * 100.0
        )

    # Filter for Monday/Tuesday Discount days
    mask_discount = (df.index.dayofweek.isin([0, 1])) & (df["Price"] < df["Week_Open"])

    signal_returns = df.loc[mask_discount, "Forward_7D_Return"].dropna()
    baseline_returns = df.loc[~mask_discount, "Forward_7D_Return"].dropna()

    if len(signal_returns) < 5 or len(baseline_returns) < 5:
        return 0.0, 1.0, 0.0, 1.0, len(signal_returns)

    t_stat, p_val_t = stats.ttest_ind(
        signal_returns, baseline_returns, equal_var=False, alternative="greater"
    )
    u_stat, p_val_u = stats.mannwhitneyu(
        signal_returns, baseline_returns, alternative="greater"
    )

    return float(t_stat), float(p_val_t), float(u_stat), float(p_val_u), len(signal_returns)


def run_daily_hypothesis_test(
    df: pd.DataFrame, lookback: int = 20, z_threshold: float = 1.0
) -> Tuple[float, float, float, float, int]:
    """Run Welch's t-test and Mann-Whitney U test for Daily Oversold Z-Score Strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing ['Price'].
    lookback : int
        Rolling window for mean and std calculation.
    z_threshold : float
        Z-score oversold boundary.

    Returns
    -------
    Tuple[float, float, float, float, int]
        (t_stat, p_value_t, u_stat, p_value_u, n_samples)
    """
    df = df.copy()
    rolling_mean = df["Price"].rolling(window=lookback).mean()
    rolling_std = df["Price"].rolling(window=lookback).std()
    df["Z_Score"] = (df["Price"] - rolling_mean) / np.maximum(rolling_std, 1e-8)

    df["Forward_1D_Return"] = (df["Price"].shift(-1) - df["Price"]) / df["Price"] * 100.0

    mask_oversold = df["Z_Score"] < -z_threshold

    signal_returns = df.loc[mask_oversold, "Forward_1D_Return"].dropna()
    baseline_returns = df.loc[~mask_oversold, "Forward_1D_Return"].dropna()

    if len(signal_returns) < 5 or len(baseline_returns) < 5:
        return 0.0, 1.0, 0.0, 1.0, len(signal_returns)

    t_stat, p_val_t = stats.ttest_ind(
        signal_returns, baseline_returns, equal_var=False, alternative="greater"
    )
    u_stat, p_val_u = stats.mannwhitneyu(
        signal_returns, baseline_returns, alternative="greater"
    )

    return float(t_stat), float(p_val_t), float(u_stat), float(p_val_u), len(signal_returns)


def evaluate_statistical_gate(
    t_stat: float, p_value: float, alpha_level: float = 0.10
) -> Dict[str, Any]:
    """Evaluate whether a trading hypothesis signal passes the statistical significance gate.

    Returns
    -------
    Dict[str, Any]
        Dictionary with boolean 'pass_gate', confidence_pct, and status message.
    """
    pass_gate = (t_stat > 0) and (p_value <= alpha_level)
    confidence = (1.0 - p_value) * 100.0

    return {
        "pass_gate": pass_gate,
        "t_stat": t_stat,
        "p_value": p_value,
        "confidence_pct": round(confidence, 2),
        "status": "Alpha Signal Statistically Significant 🟢"
        if pass_gate
        else "Failed Significance Gate (Pure Random Noise) 🔴",
    }
