"""
CryptoCast: Multi-Horizon Bitcoin Price Forecasting Using Deep Learning (PyTorch)
==================================================================================
Main entry point running the complete end-to-end pipeline using the PyTorch backend:
  Step 1: Data loading, cleaning, EDA, and preprocessing (features restricted, leak-free scaling)
  Step 2: Train all 12 models in PyTorch (1D-CNN, RNN, LSTM, Transformer for 1D, 3D, 7D horizons)
  Step 3: Generate comparison visualizations from PyTorch results

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
print("CryptoCast - PyTorch Pipeline Complete!")
print("=" * 70)
print(f"Visualizations: {os.path.join(PROJECT_DIR, 'visualizations')}")
print(f"Results Compiled: {os.path.join(PROJECT_DIR, 'model_comparison_results.csv')}")
print(f"Report: {os.path.join(PROJECT_DIR, 'CryptoCast_Project_Report.pdf')}")
