"""
CryptoCast: Multi-Horizon Bitcoin Price Forecasting Using Deep Learning
=======================================================================
Main entry point - runs the complete pipeline:
  Step 1: Data loading, cleaning, EDA, and preprocessing
  Step 2: Train all 12 models (4 architectures x 3 horizons)
  Step 3: Generate comparison visualizations

Usage:
    python cryptocast.py
"""

import os, sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from src.step1_eda import *  # noqa: F401 F403
print("\n" + "=" * 70)

from src.step2_train import *  # noqa: F401 F403
print("\n" + "=" * 70)

from src.step3_viz import *  # noqa: F401 F403

print("\n" + "=" * 70)
print("CryptoCast - Full Pipeline Complete!")
print("=" * 70)
print(f"Visualizations: {os.path.join(PROJECT_DIR, 'visualizations')}")
print(f"Results: {os.path.join(PROJECT_DIR, 'model_comparison_results.csv')}")
print(f"Report: {os.path.join(PROJECT_DIR, 'CryptoCast_Project_Report.pdf')}")
