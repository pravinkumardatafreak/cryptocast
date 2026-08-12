"""
CryptoCast - Hierarchical 2-Stage Stacking Meta-Learner Engine
===============================================================
Constructs stacked feature matrices by augmenting summary sequence features with
Stage 1 short-horizon predictions (r̂_1D, r̂_3D). Trains a Stage 2 Gradient Boosting
Meta-Learner to predict Weekly (7-Day) Directional Bias (Bullish vs Bearish).
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


def prepare_stacked_meta_features(
    raw_feature_matrix: np.ndarray,
    pred_1d_returns: np.ndarray,
    pred_3d_returns: np.ndarray,
    anchor_prices: np.ndarray,
    actual_7d_prices: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Augment original feature matrices with Stage 1 1D and 3D predicted log-returns

    and construct the binary Weekly Directional Bias target label (1 = Bullish, 0 = Bearish).

    Parameters
    ----------
    raw_feature_matrix : np.ndarray
        Shape (N, seq_len, num_features) or (N, num_features) scaled feature sequence.
    pred_1d_returns : np.ndarray
        Shape (N,) 1-day predicted log-returns from Stage 1 base model.
    pred_3d_returns : np.ndarray
        Shape (N,) 3-day predicted log-returns from Stage 1 base model.
    anchor_prices : np.ndarray
        Shape (N,) current anchor prices P_t.
    actual_7d_prices : np.ndarray, optional
        Shape (N,) actual price P_{t+7} for training labels.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X_stacked : Shape (N, num_features + 2) augmented feature matrix.
        y_weekly_bias : Binary target vector (1 if P_{t+7} > P_t else 0), or None if unlabeled.
    """
    raw_feature_matrix = np.asarray(raw_feature_matrix, dtype=np.float32)
    pred_1d_returns = np.asarray(pred_1d_returns, dtype=np.float32).reshape(-1, 1)
    pred_3d_returns = np.asarray(pred_3d_returns, dtype=np.float32).reshape(-1, 1)

    # Flatten if 3D tensor (N, seq_len, features) -> extract last step
    if raw_feature_matrix.ndim == 3:
        summary_features = raw_feature_matrix[:, -1, :]
    else:
        summary_features = raw_feature_matrix

    X_stacked = np.hstack([summary_features, pred_1d_returns, pred_3d_returns])

    y_weekly_bias = None
    if actual_7d_prices is not None:
        actual_7d_prices = np.asarray(actual_7d_prices, dtype=np.float32)
        anchor_prices = np.asarray(anchor_prices, dtype=np.float32)
        y_weekly_bias = (actual_7d_prices > anchor_prices).astype(int)

    return X_stacked, y_weekly_bias


def train_weekly_meta_classifier(
    X_train_stacked: np.ndarray,
    y_train_weekly_bias: np.ndarray,
    model_type: str = "xgboost"
):
    """Train Stage 2 Meta-Learner (XGBoost, LightGBM, or GradientBoosting) for Weekly Directional Bias prediction.

    Parameters
    ----------
    X_train_stacked : np.ndarray
        Augmented feature matrix containing raw sequence features + [r̂_1D, r̂_3D].
    y_train_weekly_bias : np.ndarray
        Binary labels (1 = Bullish, 0 = Bearish).
    model_type : str
        Choice of 'xgboost', 'lightgbm', or 'gradient_boosting'.

    Returns
    -------
    Fitted Stage 2 meta-learner model.
    """
    if model_type == "xgboost":
        try:
            import xgboost as xgb
            meta_model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=42,
                eval_metric="logloss"
            )
        except ImportError:
            meta_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    elif model_type == "lightgbm":
        try:
            import lightgbm as lgb
            meta_model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=42,
                verbosity=-1
            )
        except ImportError:
            meta_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    else:
        meta_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42,
        )

    meta_model.fit(X_train_stacked, y_train_weekly_bias)
    return meta_model


def evaluate_meta_classifier(
    meta_model: GradientBoostingClassifier,
    X_test_stacked: np.ndarray,
    y_test_weekly_bias: np.ndarray,
) -> Dict[str, float]:
    """Evaluate Stage 2 Meta-Learner performance on test set.

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy, roc_auc, and precision/recall metrics.
    """
    y_pred = meta_model.predict(X_test_stacked)
    y_prob = meta_model.predict_proba(X_test_stacked)[:, 1]

    acc = float(accuracy_score(y_test_weekly_bias, y_pred) * 100.0)
    try:
        auc = float(roc_auc_score(y_test_weekly_bias, y_prob))
    except Exception:
        auc = 0.5

    return {
        "Accuracy_Pct": round(acc, 2),
        "ROC_AUC": round(auc, 4),
    }
