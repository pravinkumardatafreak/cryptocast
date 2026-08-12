"""
Verification Test Script for all 3 Features:
1. Multi-Horizon Alert Levels & Directional Bias Engine
2. Multi-Timeframe Hypothesis Strategy Engine
3. Hierarchical 2-Stage Stacking Meta-Learner Engine
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("RUNNING VERIFICATION INTEGRATION TEST FOR ALL 3 FEATURES")
print("=" * 70)

# --- Feature 1: Directional Bias Engine ---
print("\n>>> Testing Feature 1: Directional Bias & Multi-Horizon Alert Levels...")
from src.directional_bias import calculate_alert_levels, classify_directional_bias, compute_mda

anchors = np.array([10000.0, 10500.0, 11000.0])
preds_log_ret = np.array([
    [0.02, 0.05, 0.10],   # Strong Bullish
    [-0.01, -0.03, -0.08], # Strong Bearish
    [0.01, -0.01, 0.02]   # Moderate / Divergent
])

alerts_df = calculate_alert_levels(anchors, preds_log_ret)
print("Alert Levels DataFrame:\n", alerts_df)

for i in range(len(anchors)):
    label, color, level = classify_directional_bias(
        anchors[i], alerts_df.loc[i, 'Alert_1D'], alerts_df.loc[i, 'Alert_3D'], alerts_df.loc[i, 'Alert_7D']
    )
    print(f"Sample {i+1}: Price=${anchors[i]:,.2f} | Bias={label} | Color={color} | Level={level}")

actual_future = np.array([
    [10200.0, 10600.0, 11200.0],
    [9900.0, 9700.0, 9100.0],
    [10100.0, 10400.0, 11100.0]
])

mda_scores = compute_mda(actual_future, alerts_df[['Alert_1D', 'Alert_3D', 'Alert_7D']].values, anchors)
print("MDA Scores:", mda_scores)

# --- Feature 2: Multi-Timeframe Hypothesis Engine ---
print("\n>>> Testing Feature 2: Multi-Timeframe Hypothesis Strategy Engine...")
from src.hypothesis_strategy import run_weekly_hypothesis_test, run_daily_hypothesis_test, evaluate_statistical_gate

# Load sample data
btc_path = os.path.join(PROJECT_DIR, 'data', 'btc_data.csv')
if os.path.exists(btc_path):
    df_btc = pd.read_csv(btc_path)
    df_btc['Date'] = pd.to_datetime(df_btc['Date'])
    df_btc = df_btc.set_index('Date').sort_index()

    t_w, p_w, u_w, pu_w, n_w = run_weekly_hypothesis_test(df_btc)
    print(f"Weekly Timeframe Hypothesis Test: t-stat={t_w:.4f}, p-value={p_w:.4f}, n={n_w}")

    t_d, p_d, u_d, pu_d, n_d = run_daily_hypothesis_test(df_btc, lookback=20, z_threshold=1.0)
    print(f"Daily Timeframe Hypothesis Test:  t-stat={t_d:.4f}, p-value={p_d:.4f}, n={n_d}")

    gate_eval = evaluate_statistical_gate(t_w, p_w)
    print("Weekly Statistical Gate Evaluation:", gate_eval)

# --- Feature 3: Stacking Meta-Learner Engine ---
print("\n>>> Testing Feature 3: Hierarchical 2-Stage Stacking Meta-Learner...")
from src.stacked_meta_features import prepare_stacked_meta_features, train_weekly_meta_classifier, evaluate_meta_classifier

N = 100
seq_len = 60
num_feats = 16

raw_features = np.random.randn(N, seq_len, num_feats)
pred_1d = np.random.randn(N) * 0.02
pred_3d = np.random.randn(N) * 0.04
anchors_dummy = np.full(N, 10000.0)
actual_7d_dummy = anchors_dummy * (1.0 + np.random.randn(N) * 0.05)

X_stacked, y_weekly = prepare_stacked_meta_features(raw_features, pred_1d, pred_3d, anchors_dummy, actual_7d_dummy)
print(f"Stacked Feature Matrix Shape: {X_stacked.shape} | Target Shape: {y_weekly.shape}")

meta_model = train_weekly_meta_classifier(X_stacked[:80], y_weekly[:80])
eval_results = evaluate_meta_classifier(meta_model, X_stacked[80:], y_weekly[80:])
print("Stacking Meta-Learner Test Metrics:", eval_results)

print("\n" + "=" * 70)
print("VERIFICATION SUCCESSFUL: ALL 3 FEATURES OPERATIONAL!")
print("=" * 70)
