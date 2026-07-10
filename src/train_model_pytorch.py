"""
CryptoCast - Train a single model for a single horizon using PyTorch
===================================================================
A clean, modular, and PEP-8 compliant PyTorch script that implements
1D-CNN, RNN, LSTM, and Transformer architectures for multi-horizon forecasting.

Usage:
    python src/train_model_pytorch.py <model_name> <horizon_name>
    E.g.:  python src/train_model_pytorch.py 1D-CNN 1D
"""
import os
import sys
import json
import pickle
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

warnings.filterwarnings('ignore')

# Set project directory structure
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set random seeds for reproducibility (Crucial Capstone requirement)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 64
SEQ_LENGTH = 60

# Check command line arguments
if len(sys.argv) < 3:
    print("Error: Please provide model_name and horizon_name.")
    print("Usage: python src/train_model_pytorch.py <model_name> <horizon_name>")
    sys.exit(1)

model_name = sys.argv[1]
horizon_name = sys.argv[2]

horizon_map = {'1D': 1, '3D': 3, '7D': 7}
if horizon_name not in horizon_map:
    print(f"Error: Invalid horizon {horizon_name}. Choose from 1D, 3D, 7D.")
    sys.exit(1)
horizon = horizon_map[horizon_name]

# ==========================================
# 1. DATA LOADING & SEQUENCING
# ==========================================
print(f"Loading data in PyTorch for {model_name} ({horizon_name})...")
scaled_data = np.load(os.path.join(PROJECT_DIR, 'scaled_data.npy'))

with open(os.path.join(PROJECT_DIR, 'scalers.pkl'), 'rb') as f:
    scalers = pickle.load(f)
target_scaler = scalers['target_scaler']

with open(os.path.join(PROJECT_DIR, 'meta.json'), 'r') as f:
    meta = json.load(f)
TEST_RATIO = meta['test_ratio']

def create_sequences(data_array, seq_length, horizon, target_idx=0):
    """
    Generate sliding window input sequences and matching target values.
    """
    X, y = [], []
    for i in range(len(data_array) - seq_length - horizon + 1):
        X.append(data_array[i:i + seq_length])
        y.append(data_array[i + seq_length + horizon - 1, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# Create sequences
X, y = create_sequences(scaled_data, SEQ_LENGTH, horizon, target_idx=0)
split_seq = int(len(X) * (1 - TEST_RATIO))

X_train, X_test = X[:split_seq], X[split_seq:]
y_train, y_test = y[:split_seq], y[split_seq:]

# Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train).unsqueeze(-1)  # Shape (batch, 1)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test).unsqueeze(-1)

# Chronological validation split (15% of training data)
val_split_idx = int(len(X_train_t) * 0.85)
X_tr_t, X_val_t = X_train_t[:val_split_idx], X_train_t[val_split_idx:]
y_tr_t, y_val_t = y_train_t[:val_split_idx], y_train_t[val_split_idx:]

# DataLoaders (Note: shuffle=False to prevent data leakage in time-series)
train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

input_shape = (X_train.shape[1], X_train.shape[2])  # (60, features)
input_dim = input_shape[1]

# ==========================================
# 2. PYTORCH MODEL ARCHITECTURES
# ==========================================

class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for local pattern extraction with causal padding.
    """
    def __init__(self, input_dim):
        super().__init__()
        # Causal padding is achieved by padding on left by (kernel_size - 1) and cropping the right
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=2)
        self.conv2 = nn.Conv1d(64, 64, 3, padding=2)
        self.conv3 = nn.Conv1d(64, 32, 3, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input shape: (batch, seq_len, features) -> PyTorch expects (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Causal convolutions: pad by 2 on both sides, then slice out the last 2 (right-side padding)
        x = self.relu(self.conv1(x))[:, :, :-2]
        x = self.relu(self.conv2(x))[:, :, :-2]
        x = self.relu(self.conv3(x))[:, :, :-2]
        
        x = self.pool(x).squeeze(-1)  # Shape: (batch, 32)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class RNNModel(nn.Module):
    """
    Stacked SimpleRNN baseline.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=input_dim, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.rnn2 = nn.RNN(64, 32, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.rnn1(x)
        out = self.dropout1(out)
        out, _ = self.rnn2(out)
        out = out[:, -1, :]  # Select last time step (return_sequences=False)
        out = self.dropout2(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out)


class LSTMModel(nn.Module):
    """
    Stacked LSTM model for capturing long-term temporal dependencies.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=128, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        self.dropout3 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        out = out[:, -1, :]  # Select last time step
        out = self.dropout3(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out)


class TransformerModel(nn.Module):
    """
    Transformer architecture tailored for sequence modeling with Self-Attention blocks.
    """
    def __init__(self, input_dim, head_size=64, num_heads=4, ff_dim=128, num_blocks=2):
        super().__init__()
        self.d_model = head_size * num_heads  # 256
        self.input_projection = nn.Linear(input_dim, self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.d_model, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_projection(x)  # Shape: (batch, seq_len, d_model)
        x = self.transformer_encoder(x)  # Shape: (batch, seq_len, d_model)
        x = x.transpose(1, 2)  # Shape: (batch, d_model, seq_len)
        x = self.pool(x).squeeze(-1)  # Shape: (batch, d_model)
        
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


# ==========================================
# 3. TRAINING COORDINATION
# ==========================================

# Instantiate model
if model_name == '1D-CNN':
    model = CNN1D(input_dim)
elif model_name == 'RNN':
    model = RNNModel(input_dim)
elif model_name == 'LSTM':
    model = LSTMModel(input_dim)
elif model_name == 'Transformer':
    model = TransformerModel(input_dim)
else:
    raise ValueError(f"Unknown model architecture: {model_name}")

# Criterion and Optimizer
# Use a lower learning rate for complex models (LSTM, Transformer) that are
# more sensitive to large gradient steps due to their depth and gating mechanisms.
lr_map = {'1D-CNN': 0.001, 'RNN': 0.001, 'LSTM': 0.0005, 'Transformer': 0.0005}
learning_rate = lr_map[model_name]

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning Rate Scheduler (analogous to Keras ReduceLROnPlateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-7)

# Track losses for history visualization
history = {
    'loss': [],
    'val_loss': []
}

# Early Stopping simulation variables
best_val_loss = float('inf')
best_model_weights = None
patience = 8
patience_counter = 0

print(f"Training {model_name} for {horizon_name} horizon using PyTorch...")
print(f"  Train samples={len(X_tr_t)}, Validation samples={len(X_val_t)}, Test samples={len(X_test_t)}")

for epoch in range(1, EPOCHS + 1):
    # Training step
    model.train()
    tr_loss_accum = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        tr_loss_accum += loss.item() * len(batch_x)
    
    epoch_tr_loss = tr_loss_accum / len(X_tr_t)
    
    # Validation step
    model.eval()
    val_loss_accum = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            val_loss_accum += loss.item() * len(batch_x)
    
    epoch_val_loss = val_loss_accum / len(X_val_t)
    
    # Update scheduler
    scheduler.step(epoch_val_loss)
    
    history['loss'].append(epoch_tr_loss)
    history['val_loss'].append(epoch_val_loss)
    
    # Log progress occasionally or at first/last epoch
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/{EPOCHS}: loss={epoch_tr_loss:.6f} - val_loss={epoch_val_loss:.6f}")
        
    # Check for early stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_weights = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping triggered at epoch {epoch}")
            break

# Load best weights
if best_model_weights is not None:
    model.load_state_dict(best_model_weights)

# ==========================================
# 4. EVALUATION
# ==========================================
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_t).numpy().flatten()

# Rescale predictions and targets to original price range (USD)
y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_orig = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Compute metrics
mae = mean_absolute_error(y_test_orig, y_pred_orig)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
mape = mean_absolute_percentage_error(y_test_orig, y_pred_orig) * 100

print(f"\n  PyTorch Results: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

# Save results exactly formatted as original JSON schema
result = {
    'model': model_name,
    'horizon': horizon_name,
    'MAE': float(mae),
    'RMSE': float(rmse),
    'MAPE': float(mape),
    'history': history,
    'y_test': y_test_orig.tolist(),
    'y_pred': y_pred_orig.tolist()
}

outpath = os.path.join(RESULTS_DIR, f'{model_name}_{horizon_name}.json')
with open(outpath, 'w') as f:
    json.dump(result, f)

print(f"  Saved: {outpath}\n")
