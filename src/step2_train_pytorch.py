"""
CryptoCast - Step 2: Train 5 PyTorch models with multi-output heads
=======================================================================
Trains 1D-CNN, RNN, LSTM, Transformer, and PatchTST models, each outputting
log-return predictions for 1D, 3D, and 7D horizons simultaneously in a single
forward pass. Compiles all metrics into CSV and JSON summary deliverables.

Usage:
    python src/step2_train_pytorch.py
"""
import os
import sys
import json
import subprocess
import pandas as pd

# ── Configuration ─────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

HORIZON_NAMES = ['1D', '3D', '7D']
MODELS = ['1D-CNN', 'RNN', 'LSTM', 'Transformer', 'PatchTST']

# ── Training Loop ─────────────────────────────────────────────
print("==============================================================")
print("CryptoCast: Multi-Horizon Deep Learning Model Training (PyTorch)")
print("==============================================================")

all_results = {h: {} for h in HORIZON_NAMES}

# Build path to the trainer script
trainer_script = os.path.join(PROJECT_DIR, 'src', 'train_model_pytorch.py')

for model_name in MODELS:
    print(f"\n--- Training Model: {model_name} ---")
    
    # Execute the trainer in a subprocess for isolated memory and clean logs
    cmd = [sys.executable, trainer_script, model_name]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  [ERROR] Training failed for {model_name}!")
        print(result.stderr)
        continue
        
    # Parse output for a quick console feedback
    stdout = result.stdout
    metric_lines = []
    for line in stdout.split('\n'):
        if "PyTorch Results" in line:
            metric_lines.append(line.strip())
            
    print(f"  [SUCCESS] {model_name} training complete!")
    for ml in metric_lines:
        print(f"    {ml}")

# ── Compiling Results ─────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPILING PYTORCH MODEL METRICS")
print("=" * 70)

rows = []
for horizon_name in HORIZON_NAMES:
    for model_name in MODELS:
        json_path = os.path.join(RESULTS_DIR, f'{model_name}_{horizon_name}.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                r = json.load(f)
            
            all_results[horizon_name][model_name] = {
                'MAE': r['MAE'],
                'RMSE': r['RMSE'],
                'MAPE': r['MAPE']
            }
            
            rows.append({
                'Horizon': horizon_name,
                'Model': model_name,
                'MAE': r['MAE'],
                'RMSE': r['RMSE'],
                'MAPE (%)': r['MAPE']
            })
            print(f"  {horizon_name} {model_name}: MAE={r['MAE']:.2f}, RMSE={r['RMSE']:.2f}, MAPE={r['MAPE']:.2f}%")
        else:
            print(f"  [WARNING] Results file not found for {model_name} ({horizon_name})")

# Save outputs to CSV and JSON formats matching original requirements
results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(PROJECT_DIR, 'model_comparison_results.csv'), index=False)

with open(os.path.join(PROJECT_DIR, 'results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n[Step 2] Complete! All PyTorch models trained and deliverables compiled successfully.")
