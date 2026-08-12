"""
CryptoCast - Directional Bias & Multi-Horizon Alert Engine
===========================================================
Provides mathematically rigorous functions to compute 1D, 3D, and 7D daily price
alert levels, classify directional bias vectors, and compute Mean Directional Accuracy (MDA).
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def calculate_alert_levels(
    anchor_prices: np.ndarray, log_returns_pred: np.ndarray
) -> pd.DataFrame:
    """Calculate 1D, 3D, and 7D price alert levels from predicted log-returns.

    Formula:
        P_{t+h} = P_t * exp(r_{t, h})

    Parameters
    ----------
    anchor_prices : np.ndarray
        Shape (N,) array of current close prices P_t at each daily step t.
    log_returns_pred : np.ndarray
        Shape (N, 3) matrix containing predicted log returns [1D, 3D, 7D].

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Anchor_Price', 'Alert_1D', 'Alert_3D', 'Alert_7D'].
    """
    anchor_prices = np.asarray(anchor_prices, dtype=np.float64)
    log_returns_pred = np.asarray(log_returns_pred, dtype=np.float64)

    alert_1d = anchor_prices * np.exp(log_returns_pred[:, 0])
    alert_3d = anchor_prices * np.exp(log_returns_pred[:, 1])
    alert_7d = anchor_prices * np.exp(log_returns_pred[:, 2])

    return pd.DataFrame(
        {
            "Anchor_Price": anchor_prices,
            "Alert_1D": alert_1d,
            "Alert_3D": alert_3d,
            "Alert_7D": alert_7d,
        }
    )


def classify_directional_bias(
    anchor_price: float, alert_1d: float, alert_3d: float, alert_7d: float
) -> Tuple[str, str, int]:
    """Classify daily directional bias based on vector alignment between price

    and 3 alert levels.

    Rules:
    - Strong Bullish (+2): P_t < 1D < 3D < 7D (Ascending expansion)
    - Strong Bearish (-2): P_t > 1D > 3D > 7D (Descending contraction)
    - Moderate Bullish (+1): Majority positive predicted returns
    - Moderate Bearish (-1): Majority negative predicted returns
    - Neutral / Divergent (0): Mixed conflicting signals

    Returns
    -------
    Tuple[str, str, int]
        (bias_label, alert_color_hex, numeric_bias_level)
    """
    ret_1d = (alert_1d - anchor_price) / anchor_price
    ret_3d = (alert_3d - anchor_price) / anchor_price
    ret_7d = (alert_7d - anchor_price) / anchor_price

    # Strong Bullish Alignment: P_t < 1D < 3D < 7D
    if anchor_price < alert_1d < alert_3d < alert_7d:
        return "Strong Bullish 🟢", "#22c55e", 2

    # Strong Bearish Alignment: P_t > 1D > 3D > 7D
    if anchor_price > alert_1d > alert_3d > alert_7d:
        return "Strong Bearish 🔴", "#ef4444", -2

    positive_count = sum([r > 0 for r in (ret_1d, ret_3d, ret_7d)])

    if positive_count >= 2:
        return "Moderate Bullish 🟢", "#4ade80", 1
    elif positive_count == 0 or (ret_1d < 0 and ret_7d < 0):
        return "Moderate Bearish 🟠", "#f97316", -1
    else:
        return "Neutral / Divergent ⚪", "#94a3b8", 0


def compute_mda(
    actual_future_prices: np.ndarray,
    predicted_alert_prices: np.ndarray,
    anchor_prices: np.ndarray,
) -> Dict[str, float]:
    """Compute Mean Directional Accuracy (MDA) across 1D, 3D, and 7D horizons.

    Formula:
        MDA_h = (1/N) * sum( sign(P_{t+h} - P_t) == sign(P̂_{t+h} - P_t) ) * 100%

    Parameters
    ----------
    actual_future_prices : np.ndarray
        Shape (N, 3) matrix containing actual future prices [1D, 3D, 7D].
    predicted_alert_prices : np.ndarray
        Shape (N, 3) matrix containing predicted alert prices [1D, 3D, 7D].
    anchor_prices : np.ndarray
        Shape (N,) array of anchor close prices P_t.

    Returns
    -------
    Dict[str, float]
        Dictionary with MDA percentages for '1D', '3D', and '7D'.
    """
    actual_future_prices = np.asarray(actual_future_prices, dtype=np.float64)
    predicted_alert_prices = np.asarray(predicted_alert_prices, dtype=np.float64)
    anchor_prices = np.asarray(anchor_prices, dtype=np.float64).reshape(-1, 1)

    actual_directions = np.sign(actual_future_prices - anchor_prices)
    pred_directions = np.sign(predicted_alert_prices - anchor_prices)

    horizons = ["1D", "3D", "7D"]
    mda_results = {}

    for idx, h in enumerate(horizons):
        match_vector = actual_directions[:, idx] == pred_directions[:, idx]
        mda_score = float(np.mean(match_vector) * 100.0)
        mda_results[f"MDA_{h}"] = round(mda_score, 2)

    return mda_results
