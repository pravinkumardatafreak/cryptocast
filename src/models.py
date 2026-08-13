"""
CryptoCast - Centralized PyTorch Model Architectures & Custom Loss
===================================================================
A clean, modular, and PEP-8 compliant module containing all multi-output
PyTorch neural network architectures (1D-CNN, RNN, LSTM, Transformer, PatchTST)
and custom loss functions (DirectionalMSELoss) to eliminate code duplication.
"""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for local pattern extraction with causal padding.
    Outputs predictions for 1D, 3D, and 7D horizons simultaneously.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=2)
        self.conv2 = nn.Conv1d(64, 64, 3, padding=2)
        self.conv3 = nn.Conv1d(64, 32, 3, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 3)  # Multi-output output layer (1D, 3D, 7D)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.relu(self.conv1(x))[:, :, :-2]
        x = self.relu(self.conv2(x))[:, :, :-2]
        x = self.relu(self.conv3(x))[:, :, :-2]
        
        x = self.pool(x).squeeze(-1)  # (batch, 32)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class RNNModel(nn.Module):
    """
    Stacked SimpleRNN baseline for sequence modeling.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=input_dim, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.rnn2 = nn.RNN(64, 32, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 3)  # Multi-output output layer
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn1(x)
        out = self.dropout1(out)
        out, _ = self.rnn2(out)
        out = out[:, -1, :]  # Select last time step
        out = self.dropout2(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out)


class LSTMModel(nn.Module):
    """
    Stacked LSTM model for capturing long-term temporal dependencies.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=128, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        self.dropout3 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 3)  # Multi-output output layer
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    Transformer architecture tailored for sequence modeling with Multi-Head Self-Attention.
    """
    def __init__(self, input_dim: int, head_size: int = 64, num_heads: int = 4, ff_dim: int = 128, num_blocks: int = 2):
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
        self.fc2 = nn.Linear(64, 3)  # Multi-output output layer
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)
        
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


class RevIN(nn.Module):
    """
    Reversible Instance Normalization for solving distribution shift in Time Series.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = x - self.mean
            x = x / self.stdev
            x = x * self.affine + self.beta
            return x
        elif mode == 'denorm':
            x = x - self.beta[0]
            x = x / self.affine[0]
            x = x * self.stdev[:, :, 0]
            x = x + self.mean[:, :, 0]
            return x


class PatchTSTModel(nn.Module):
    """
    State-of-the-art Patch Time Series Transformer with RevIN.
    Groups consecutive timesteps into patches to reduce self-attention complexity.
    """
    def __init__(self, input_dim: int, seq_len: int = 60, patch_len: int = 12, stride: int = 12, d_model: int = 128, n_heads: int = 4, e_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.revin = RevIN(input_dim)
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = int((seq_len - patch_len) / stride + 1)
        
        self.value_embedding = nn.Linear(patch_len * input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.empty(1, self.patch_num, d_model))
        nn.init.uniform_(self.position_embedding, -0.1, 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=256,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.patch_num * d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.revin(x, 'norm')
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.reshape(patches.shape[0], patches.shape[1], -1)
        x = self.value_embedding(patches) + self.position_embedding
        x = self.encoder(x)
        x = self.head(x)
        return x


class DirectionalMSELoss(nn.Module):
    """
    Custom Loss Function penalizing directional disagreement (sign penalty)
    in addition to Mean Squared Error.
    """
    def __init__(self, alpha: float = 0.30):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(y_pred, y_true)
        true_sign = torch.sign(y_true)
        dir_penalty = torch.mean(torch.relu(-y_pred * true_sign))
        return mse_loss + (self.alpha * dir_penalty)


def get_model_dict(input_dim: int):
    """Returns a dictionary mapping model names to instantiated PyTorch model objects."""
    return {
        '1D-CNN': CNN1D(input_dim),
        'RNN': RNNModel(input_dim),
        'LSTM': LSTMModel(input_dim),
        'Transformer': TransformerModel(input_dim),
        'PatchTST': PatchTSTModel(input_dim)
    }
