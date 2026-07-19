"""
CryptoCast: Multi-Horizon Bitcoin Price Forecasting Using Deep Learning (PyTorch)
==================================================================================
Main entry point running the complete end-to-end pipeline using the PyTorch backend:
  Step 1: Data loading, cleaning, EDA, and preprocessing (9 features, leak-free scaling)
  Step 2: Train 5 models (1D-CNN, RNN, LSTM, Transformer, PatchTST) each predicting 1D, 3D, 7D
          horizons simultaneously via multi-output architecture
  Step 3: Generate comparison visualizations from results

Usage:
    python cryptocast.py
"""

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

print("\n" + "=" * 70)
print("CRYPTOCAST PIPELINE (PYTORCH EDITION)")
print("=" * 70)

# Step 1: Preprocessing and EDA
print("\n>>> Launching Step 1: Exploratory Data Analysis & Sequence Preprocessing...")
from src.step1_eda import *  # noqa: F401 F403
print("\n" + "=" * 70)

# Step 2: Training PyTorch Models
print("\n>>> Launching Step 2: Deep Learning Model Training (PyTorch Backend)...")
from src.step2_train_pytorch import *  # noqa: F401 F403
print("\n" + "=" * 70)

# Step 3: Visualization Generation
print("\n>>> Launching Step 3: Visual Comparison & Evaluation Charts...")
from src.step3_viz import *  # noqa: F401 F403
print("\n" + "=" * 70)

# Step 4: Walk-Forward Validation (Backtesting)
print("\n>>> Launching Step 4: Walk-Forward Validation (Backtesting)...")
import subprocess
wfv_script = os.path.join(PROJECT_DIR, 'src', 'step4_wfv.py')
result = subprocess.run([sys.executable, wfv_script])
if result.returncode != 0:
    print("  [ERROR] Walk-Forward Validation failed!")
else:
    print("  [SUCCESS] WFV Backtesting complete!")

print("\n" + "=" * 70)
print("CryptoCast - PyTorch Pipeline Complete!")
print("=" * 70)
print(f"Visualizations: {os.path.join(PROJECT_DIR, 'visualizations')}")
print(f"Results Compiled: {os.path.join(PROJECT_DIR, 'model_comparison_results.csv')}")
print(f"WFV Backtest: {os.path.join(PROJECT_DIR, 'wfv_results.json')}")

