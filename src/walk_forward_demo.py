"""
CryptoCast - Walk-Forward (Expanding Window) Validation Demo
============================================================
Demonstrates the walk-forward validation methodology for time-series forecasting.
This script implements an expanding training window:
  - Iteration 1: Train on first 450 days, predict the next 90 days.
  - Iteration 2: Train on first 540 days, predict the next 90 days.
  - ... and so on, sliding forward by 90 days at each step.

A fast Linear Regression baseline is used to showcase this logic instantly
without the massive training overhead of 12 deep learning models over 50 folds.

Usage:
    python src/walk_forward_demo.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'btc_data.csv')

def main():
    print("=" * 70)
    print("WALK-FORWARD (EXPANDING WINDOW) VALIDATION DEMO")
    print("=" * 70)
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Please run step1_eda.py first.")
        return
        
    # Load raw data
    data = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
    # We restrict features to Open and Close (Price) as per guidelines
    features = ['Price', 'Open']
    df = data[features].copy()
    
    # We want to use a sliding window of 60 days to predict the next price (1D horizon)
    seq_length = 60
    
    # Generate lagged features for linear regression model (X: past 60 days Close & Open, y: next Close)
    # Create lag features for close (Price) and open
    X_list, y_list = [], []
    for i in range(len(df) - seq_length):
        lag_window = df.iloc[i : i + seq_length].values.flatten() # 120 features (60 Close + 60 Open)
        target = df.iloc[i + seq_length]['Price']
        X_list.append(lag_window)
        y_list.append(target)
        
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Walk-forward parameters
    initial_train_size = 450
    test_size = 90
    total_samples = len(X)
    
    # Walk-forward loop
    fold = 1
    start_train = 0
    fold_metrics = []
    
    print(f"Total sequences generated: {total_samples}")
    print(f"Initial train size: {initial_train_size} sequences")
    print(f"Test step size: {test_size} sequences")
    print("-" * 70)
    print(f"{'Fold':<6} | {'Train Range':<15} | {'Test Range':<15} | {'MAE (USD)':<10} | {'RMSE (USD)':<10} | {'MAPE (%)':<8}")
    print("-" * 70)
    
    current_train_end = initial_train_size
    
    while current_train_end + test_size <= total_samples:
        # Split features and target
        X_train, X_test = X[start_train:current_train_end], X[current_train_end:current_train_end + test_size]
        y_train, y_test = y[start_train:current_train_end], y[current_train_end:current_train_end + test_size]
        
        # Fit scaler ONLY on the train portion of this fold (Avoid Data Leakage!)
        scaler_X = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        # Scale target
        scaler_y = MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Train model (Linear Regression)
        model = LinearRegression()
        model.fit(X_train_scaled, y_train_scaled)
        
        # Predict
        preds_scaled = model.predict(X_test_scaled)
        # Inverse scale predictions
        preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        
        # Compute metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = mean_absolute_percentage_error(y_test, preds) * 100
        
        fold_metrics.append({
            'fold': fold, 'mae': mae, 'rmse': rmse, 'mape': mape
        })
        
        print(f"{fold:<6} | {start_train:04d} to {current_train_end:04d} | {current_train_end:04d} to {current_train_end + test_size:04d} | {mae:<10.2f} | {rmse:<10.2f} | {mape:<8.2f}%")
        
        # Expand training set to include this fold's test set
        current_train_end += test_size
        fold += 1
        
    print("-" * 70)
    # Calculate average metrics
    avg_mae = np.mean([m['mae'] for m in fold_metrics])
    avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
    avg_mape = np.mean([m['mape'] for m in fold_metrics])
    
    print(f"Average Walk-Forward Performance (across {fold-1} folds):")
    print(f"  Mean MAE:  ${avg_mae:,.2f}")
    print(f"  Mean RMSE: ${avg_rmse:,.2f}")
    print(f"  Mean MAPE: {avg_mape:.2f}%")
    print("=" * 70)
    print("Viva Tip: Walk-forward validation provides a realistic assessment of financial models")
    print("          because it prevents future information from leaking into model training,")
    print("          mimicking how a trading bot would retrain as new data arrives.")
    print("=" * 70)

if __name__ == '__main__':
    main()
