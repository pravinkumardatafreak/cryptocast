"""
CryptoCast - Step 4: Walk-Forward Validation (Backtesting)
===========================================================
Implements a 3-Fold Expanding Window Walk-Forward Validation to evaluate model
robustness across different market regimes without temporal data leakage.

Usage:
    python src/step4_wfv.py
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'btc_data.csv')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

EPOCHS = 20  # Keep epochs low for fast multi-fold training
BATCH_SIZE = 64
SEQ_LENGTH = 60
HORIZON_NAMES = ['1D', '3D', '7D']
MODELS = ['RNN', '1D-CNN', 'LSTM', 'Transformer']

# ── Model Architectures (Multi-Output PyTorch) ─────────────────
class CNN1D(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=2)
        self.conv2 = nn.Conv1d(64, 64, 3, padding=2)
        self.conv3 = nn.Conv1d(64, 32, 3, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))[:, :, :-2]
        x = self.relu(self.conv2(x))[:, :, :-2]
        x = self.relu(self.conv3(x))[:, :, :-2]
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class RNNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.rnn1 = nn.RNN(input_dim, 64, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.rnn2 = nn.RNN(64, 32, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.rnn1(x)
        out = self.dropout1(out)
        out, _ = self.rnn2(out)
        out = out[:, -1, :]
        out = self.dropout2(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out)

class LSTMModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, 128, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        self.dropout3 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        out = out[:, -1, :]
        out = self.dropout3(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, head_size=64, num_heads=4, ff_dim=128, num_blocks=2):
        super().__init__()
        self.d_model = head_size * num_heads
        self.input_projection = nn.Linear(input_dim, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=0.1, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.d_model, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

# ── Data Processing Helpers ────────────────────────────────────
def create_sequences_multi(scaled_features, raw_prices, seq_length=60):
    X, y, anchors, actuals = [], [], [], []
    for i in range(len(scaled_features) - seq_length - 7 + 1):
        X.append(scaled_features[i : i + seq_length])
        anchor_p = raw_prices[i + seq_length - 1]
        anchors.append(anchor_p)
        
        p_1d = raw_prices[i + seq_length]
        p_3d = raw_prices[i + seq_length + 2]
        p_7d = raw_prices[i + seq_length + 6]
        
        r_1d = np.log(p_1d / anchor_p)
        r_3d = np.log(p_3d / anchor_p)
        r_7d = np.log(p_7d / anchor_p)
        
        y.append([r_1d, r_3d, r_7d])
        actuals.append([p_1d, p_3d, p_7d])
        
    return (np.array(X, dtype=np.float32), 
            np.array(y, dtype=np.float32), 
            np.array(anchors, dtype=np.float32), 
            np.array(actuals, dtype=np.float32))

# ── Main WFV Pipeline ──────────────────────────────────────────
def main():
    print("==============================================================")
    # Use standard characters for stability
    print("CryptoCast: Walk-Forward Validation (Backtesting)")
    print("==============================================================")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at {DATA_PATH}. Run step1_eda.py first.")
        
    data = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
    features = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %', 'SMA_7', 'SMA_30', 'RSI_14', 'Vol_30']
    raw_prices = data['Price'].values
    
    # 3-Fold Expanding Window boundaries
    # We will split on the total sequence length to align indices exactly
    M = len(data) - SEQ_LENGTH - 7 + 1
    
    folds = [
        {
            'train_end': int(M * 0.60),
            'test_end': int(M * 0.7333),
            'desc': 'Fold 1: 2010-2018 (Train) -> 2019-2020 (Test)'
        },
        {
            'train_end': int(M * 0.7333),
            'test_end': int(M * 0.8666),
            'desc': 'Fold 2: 2010-2020 (Train) -> 2020-2022 (Test)'
        },
        {
            'train_end': int(M * 0.8666),
            'test_end': M,
            'desc': 'Fold 3: 2010-2022 (Train) -> 2022-2024 (Test)'
        }
    ]
    
    wfv_results = {}
    
    for model_name in MODELS:
        print(f"\n--- Model Architecture: {model_name} ---")
        wfv_results[model_name] = []
        
        for fold_idx, fold in enumerate(folds, 1):
            print(f"  Running Fold {fold_idx}/3: {fold['desc']}")
            
            # 1. Feature scaling strictly fit on train partition to avoid leakage
            train_idx_end = fold['train_end']
            test_idx_end = fold['test_end']
            
            scaler = MinMaxScaler()
            # Fit only on the training subset of dataframe features
            # The indices for sequences map to the raw data index offset by SEQ_LENGTH
            raw_train_end = train_idx_end + SEQ_LENGTH
            scaler.fit(data[features].iloc[:raw_train_end])
            
            scaled_data = scaler.transform(data[features])
            
            # Generate target sequences
            X_all, y_all, anchors_all, actuals_all = create_sequences_multi(scaled_data, raw_prices, SEQ_LENGTH)
            
            # Splitting subsets chronologically
            X_train, X_test = X_all[:train_idx_end], X_all[train_idx_end:test_idx_end]
            y_train, y_test = y_all[:train_idx_end], y_all[train_idx_end:test_idx_end]
            anchors_train, anchors_test = anchors_all[:train_idx_end], anchors_all[train_idx_end:test_idx_end]
            actuals_train, actuals_test = actuals_all[:train_idx_end], actuals_all[train_idx_end:test_idx_end]
            
            # Chronological validation split (15% of training partition)
            val_split = int(len(X_train) * 0.85)
            X_tr, X_val = X_train[:val_split], X_train[val_split:]
            y_tr, y_val = y_train[:val_split], y_train[val_split:]
            
            # DataLoaders
            train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=BATCH_SIZE, shuffle=False)
            val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE, shuffle=False)
            
            # Setup Model
            input_dim = len(features)
            if model_name == 'RNN':
                model = RNNModel(input_dim)
            elif model_name == '1D-CNN':
                model = CNN1D(input_dim)
            elif model_name == 'LSTM':
                model = LSTMModel(input_dim)
            elif model_name == 'Transformer':
                model = TransformerModel(input_dim)
                
            criterion = nn.MSELoss()
            lr_map = {'RNN': 0.001, '1D-CNN': 0.001, 'LSTM': 0.0005, 'Transformer': 0.0005}
            optimizer = optim.Adam(model.parameters(), lr=lr_map[model_name])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
            
            # Training loop with early stopping
            best_val_loss = float('inf')
            best_weights = None
            patience = 5
            patience_counter = 0
            
            for epoch in range(1, EPOCHS + 1):
                model.train()
                for bx, by in train_loader:
                    optimizer.zero_grad()
                    preds = model(bx)
                    loss = criterion(preds, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        preds = model(bx)
                        loss = criterion(preds, by)
                        val_loss += loss.item() * len(bx)
                val_loss /= len(X_val)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
                        
            if best_weights is not None:
                model.load_state_dict(best_weights)
                
            # Predict
            model.eval()
            with torch.no_grad():
                y_pred_returns = model(torch.tensor(X_test)).numpy()
                
            # Reconstruct prices: P_t+h = P_t * exp(r_h)
            pred_prices = np.zeros_like(actuals_test)
            pred_prices[:, 0] = anchors_test * np.exp(y_pred_returns[:, 0])
            pred_prices[:, 1] = anchors_test * np.exp(y_pred_returns[:, 1])
            pred_prices[:, 2] = anchors_test * np.exp(y_pred_returns[:, 2])
            
            fold_metrics = {}
            for h_idx, h_name in enumerate(HORIZON_NAMES):
                y_true_h = actuals_test[:, h_idx]
                y_pred_h = pred_prices[:, h_idx]
                
                mae = mean_absolute_error(y_true_h, y_pred_h)
                rmse = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
                mape = mean_absolute_percentage_error(y_true_h, y_pred_h) * 100
                
                fold_metrics[h_name] = {
                    'MAE': float(mae),
                    'RMSE': float(rmse),
                    'MAPE': float(mape)
                }
                
            print(f"    1D MAPE: {fold_metrics['1D']['MAPE']:.2f}% | 3D MAPE: {fold_metrics['3D']['MAPE']:.2f}% | 7D MAPE: {fold_metrics['7D']['MAPE']:.2f}%")
            wfv_results[model_name].append(fold_metrics)
            
    # Save validation metrics summary
    out_path = os.path.join(PROJECT_DIR, 'wfv_results.json')
    with open(out_path, 'w') as f:
        json.dump(wfv_results, f, indent=2)
    print(f"\n[Step 4] Complete! Saved WFV metrics to: {out_path}")

if __name__ == '__main__':
    main()
