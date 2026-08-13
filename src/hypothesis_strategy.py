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
    """Run Welch's t-test and Mann-Whitney U test for Technical Mon/Tue Discount Days (Without AI)."""
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

    # Filter for Monday/Tuesday Discount days (Technical ONLY)
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


def run_ai_dual_directional_hypothesis_test(
    df: pd.DataFrame, ai_preds_dict: Dict[str, np.ndarray] = None
) -> Tuple[float, float, float, float, pd.Series, pd.Series]:
    """Run Welch's t-test & Mann-Whitney U test for the Dual-Directional AI Strategy.

    Evaluates:
      - LONG signals: Mon/Tue & Price < Week_Open & AI Bullish (Forward 7D Long Return)
      - SHORT signals: Mon/Tue & Price > Week_Open & AI Bearish (Forward 7D Short Return)

    Returns
    -------
    Tuple[t_stat, p_val_t, u_stat, p_val_u, ai_strategy_returns, baseline_returns]
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

    df["Forward_7D_Long_Return"] = (df["Price"].shift(-7) - df["Price"]) / df["Price"] * 100.0
    df["Forward_7D_Short_Return"] = (df["Price"] - df["Price"].shift(-7)) / df["Price"] * 100.0

    is_mon_tue = df.index.dayofweek.isin([0, 1])
    is_discount = df["Price"] < df["Week_Open"]
    is_premium = df["Price"] > df["Week_Open"]

    # Check if AI predictions provided
    if ai_preds_dict and len(ai_preds_dict) > 0:
        # Align predictions to dataframe end
        min_len = min([len(v) for v in ai_preds_dict.values()])
        eval_df = df.iloc[-min_len:].copy()

        # Compute AI consensus return forecast (7D horizon)
        preds_7d = [ai_preds_dict[m][-min_len:, 2] if ai_preds_dict[m].ndim == 2 else ai_preds_dict[m][-min_len:] for m in ai_preds_dict]
        avg_pred_7d = np.mean(preds_7d, axis=0)

        ai_bullish = avg_pred_7d > 0
        ai_bearish = avg_pred_7d < 0

        mon_tue_eval = eval_df.index.dayofweek.isin([0, 1])
        discount_eval = eval_df["Price"] < eval_df["Week_Open"]
        premium_eval = eval_df["Price"] > eval_df["Week_Open"]

        mask_long = mon_tue_eval & discount_eval & ai_bullish
        mask_short = mon_tue_eval & premium_eval & ai_bearish

        long_returns = eval_df.loc[mask_long, "Forward_7D_Long_Return"].dropna()
        short_returns = eval_df.loc[mask_short, "Forward_7D_Short_Return"].dropna()
        strategy_returns = pd.concat([long_returns, short_returns])

        mask_baseline = ~(mask_long | mask_short)
        baseline_returns = eval_df.loc[mask_baseline, "Forward_7D_Long_Return"].dropna()
    else:
        # Fallback heuristic alignment
        mask_long = is_mon_tue & is_discount
        mask_short = is_mon_tue & is_premium
        long_returns = df.loc[mask_long, "Forward_7D_Long_Return"].dropna()
        short_returns = df.loc[mask_short, "Forward_7D_Short_Return"].dropna()
        strategy_returns = pd.concat([long_returns, short_returns])
        baseline_returns = df.loc[~(mask_long | mask_short), "Forward_7D_Long_Return"].dropna()

    if len(strategy_returns) < 5 or len(baseline_returns) < 5:
        return 0.0, 1.0, 0.0, 1.0, strategy_returns, baseline_returns

    t_stat, p_val_t = stats.ttest_ind(
        strategy_returns, baseline_returns, equal_var=False, alternative="greater"
    )
    u_stat, p_val_u = stats.mannwhitneyu(
        strategy_returns, baseline_returns, alternative="greater"
    )

    return float(t_stat), float(p_val_t), float(u_stat), float(p_val_u), strategy_returns, baseline_returns


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
