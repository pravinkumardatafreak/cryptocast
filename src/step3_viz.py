"""
CryptoCast - Step 3: Generate comparison visualizations
=======================================================
Reads training results from JSON files and creates all comparison charts.

Usage:
    python src/step3_viz.py
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Configuration ─────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIZ_DIR = os.path.join(PROJECT_DIR, 'visualizations')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

model_names = ['1D-CNN', 'RNN', 'LSTM', 'Transformer']
horizon_names = ['1D', '3D', '7D']
model_colors = {'1D-CNN': '#3498db', 'RNN': '#e74c3c', 'LSTM': '#2ecc71', 'Transformer': '#9b59b6'}

# ── Load results ──────────────────────────────────────────────
all_results = {}
all_histories = {}
all_predictions = {}

for horizon_name in horizon_names:
    all_results[horizon_name] = {}
    all_histories[horizon_name] = {}
    all_predictions[horizon_name] = {}
    for model_name in model_names:
        fpath = os.path.join(RESULTS_DIR, f'{model_name}_{horizon_name}.json')
        with open(fpath, 'r') as f:
            data = json.load(f)
        all_results[horizon_name][model_name] = {
            'MAE': data['MAE'], 'RMSE': data['RMSE'], 'MAPE': data['MAPE']
        }
        all_histories[horizon_name][model_name] = data['history']
        all_predictions[horizon_name][model_name] = {
            'y_test': data['y_test'], 'y_pred': data['y_pred']
        }

# Save results CSV
rows = []
for horizon_name in horizon_names:
    for model_name in model_names:
        r = all_results[horizon_name][model_name]
        rows.append({'Horizon': horizon_name, 'Model': model_name, 'MAE': r['MAE'], 'RMSE': r['RMSE'], 'MAPE (%)': r['MAPE']})
results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(PROJECT_DIR, 'model_comparison_results.csv'), index=False)

with open(os.path.join(PROJECT_DIR, 'results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

print("=" * 70)
print("MODEL COMPARISON SUMMARY")
print("=" * 70)
print(results_df.to_string(index=False))
print("\n--- Best Model per Horizon ---")
for horizon_name in horizon_names:
    hdf = results_df[results_df['Horizon'] == horizon_name]
    best = hdf.loc[hdf['MAE'].idxmin()]
    print(f"  {horizon_name}: {best['Model']} (MAE={best['MAE']:.2f})")

# ── Visualizations ────────────────────────────────────────────
print("\nGenerating visualizations...")

# 09 - Loss Curves
for horizon_name in horizon_names:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, model_name in enumerate(model_names):
        hist = all_histories[horizon_name][model_name]
        ax = axes[idx]
        ax.plot(hist['loss'], label='Train Loss', color=model_colors[model_name], linewidth=1.5)
        ax.plot(hist['val_loss'], label='Val Loss', color='#e67e22', linewidth=1.5, linestyle='--')
        ax.set_title(f'{model_name} - {horizon_name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (MSE)')
        ax.legend(loc='best'); ax.grid(True, alpha=0.3)
    fig.suptitle(f'Training Loss Curves - {horizon_name} Horizon', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f'09_loss_curves_{horizon_name}.png'), bbox_inches='tight')
    plt.close()
print("  Loss curves saved")

# 10 - Actual vs Predicted
for horizon_name in horizon_names:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for idx, model_name in enumerate(model_names):
        preds = all_predictions[horizon_name][model_name]
        ax = axes[idx]
        ax.plot(preds['y_test'], label='Actual', color='#1a5276', linewidth=1.2, alpha=0.9)
        ax.plot(preds['y_pred'], label='Predicted', color=model_colors[model_name], linewidth=1.2, alpha=0.8)
        ax.set_title(model_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Test Sample Index'); ax.set_ylabel('Price (USD)')
        ax.legend(loc='best'); ax.grid(True, alpha=0.3)
    fig.suptitle(f'Actual vs Predicted - {horizon_name} Horizon', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f'10_actual_vs_predicted_{horizon_name}.png'), bbox_inches='tight')
    plt.close()
print("  Actual vs Predicted plots saved")

# 11 - Error Distribution
for horizon_name in horizon_names:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, model_name in enumerate(model_names):
        preds = all_predictions[horizon_name][model_name]
        errors = np.array(preds['y_pred']) - np.array(preds['y_test'])
        ax = axes[idx]
        ax.hist(errors, bins=50, color=model_colors[model_name], alpha=0.7, edgecolor='white')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.axvline(x=np.mean(errors), color='red', linestyle='-', linewidth=1.5, label=f'Mean={np.mean(errors):.0f}')
        ax.set_title(model_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Prediction Error (USD)'); ax.set_ylabel('Frequency')
        ax.legend(loc='best'); ax.grid(True, alpha=0.3)
    fig.suptitle(f'Error Distribution - {horizon_name} Horizon', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f'11_error_distribution_{horizon_name}.png'), bbox_inches='tight')
    plt.close()
print("  Error distribution plots saved")

# 12 - MAE Bar Charts
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, horizon_name in enumerate(horizon_names):
    hdf = results_df[results_df['Horizon'] == horizon_name]
    bars = axes[idx].bar(hdf['Model'], hdf['MAE'], color=[model_colors[m] for m in hdf['Model']], edgecolor='white', linewidth=1.5)
    axes[idx].set_title(f'{horizon_name} Horizon', fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('MAE (USD)'); axes[idx].grid(True, alpha=0.3, axis='y')
    for bar in bars:
        axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
fig.suptitle('Model Comparison by MAE', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '12_mae_comparison.png'), bbox_inches='tight')
plt.close()
print("  MAE comparison saved")

# 13 - RMSE Bar Charts
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, horizon_name in enumerate(horizon_names):
    hdf = results_df[results_df['Horizon'] == horizon_name]
    bars = axes[idx].bar(hdf['Model'], hdf['RMSE'], color=[model_colors[m] for m in hdf['Model']], edgecolor='white', linewidth=1.5)
    axes[idx].set_title(f'{horizon_name} Horizon', fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('RMSE (USD)'); axes[idx].grid(True, alpha=0.3, axis='y')
    for bar in bars:
        axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
fig.suptitle('Model Comparison by RMSE', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '13_rmse_comparison.png'), bbox_inches='tight')
plt.close()
print("  RMSE comparison saved")

# 14 - MAPE Bar Charts
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, horizon_name in enumerate(horizon_names):
    hdf = results_df[results_df['Horizon'] == horizon_name]
    bars = axes[idx].bar(hdf['Model'], hdf['MAPE (%)'], color=[model_colors[m] for m in hdf['Model']], edgecolor='white', linewidth=1.5)
    axes[idx].set_title(f'{horizon_name} Horizon', fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('MAPE (%)'); axes[idx].grid(True, alpha=0.3, axis='y')
    for bar in bars:
        axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
fig.suptitle('Model Comparison by MAPE', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '14_mape_comparison.png'), bbox_inches='tight')
plt.close()
print("  MAPE comparison saved")

# 15 - Grouped Horizon Comparison
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(model_names))
width = 0.25
for i, horizon_name in enumerate(horizon_names):
    hdf = results_df[results_df['Horizon'] == horizon_name]
    vals = hdf['MAE'].values
    bars = ax.bar(x + i * width, vals, width, label=horizon_name, alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)
ax.set_xlabel('Model', fontsize=13); ax.set_ylabel('MAE (USD)', fontsize=13)
ax.set_title('Multi-Horizon MAE Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x + width); ax.set_xticklabels(model_names, fontsize=12)
ax.legend(loc='best'); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '15_horizon_comparison.png'), bbox_inches='tight')
plt.close()
print("  Horizon comparison saved")

# 16 - Scatter: Actual vs Predicted (best per horizon)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, horizon_name in enumerate(horizon_names):
    hdf = results_df[results_df['Horizon'] == horizon_name]
    best_model = hdf.loc[hdf['MAE'].idxmin(), 'Model']
    preds = all_predictions[horizon_name][best_model]
    y_test_arr = np.array(preds['y_test'])
    y_pred_arr = np.array(preds['y_pred'])
    ax = axes[idx]
    ax.scatter(y_test_arr, y_pred_arr, alpha=0.4, s=15, color=model_colors[best_model], edgecolors='white', linewidth=0.3)
    min_val = min(y_test_arr.min(), y_pred_arr.min())
    max_val = max(y_test_arr.max(), y_pred_arr.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Perfect')
    ax.set_title(f'{horizon_name} - Best: {best_model}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Actual Price (USD)'); ax.set_ylabel('Predicted Price (USD)')
    ax.legend(loc='best'); ax.grid(True, alpha=0.3)
fig.suptitle('Scatter: Actual vs Predicted (Best Model per Horizon)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '16_scatter_actual_vs_pred.png'), bbox_inches='tight')
plt.close()
print("  Scatter plots saved")

print("\n[Step 3] Complete! All visualizations generated.")
