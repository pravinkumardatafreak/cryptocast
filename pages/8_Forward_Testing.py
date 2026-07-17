import streamlit as st
import os
import sys
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Set page config
st.set_page_config(
    page_title="CryptoCast | True OOS Forward Testing",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.streamlit_utils import inject_custom_css, card, callout, DARK_LAYOUT
inject_custom_css()

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
SCALER_PATH = os.path.join(PROJECT_DIR, 'scalers.pkl')
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'btc_data.csv')

# ==============================================================================
# Model Architectures (Must exactly match train_model_pytorch.py)
# ==============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        return self.fc2(out)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, head_size=64, num_heads=4, ff_dim=128, num_blocks=2):
        super().__init__()
        self.d_model = head_size * num_heads
        self.input_projection = nn.Linear(input_dim, self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=0.0, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(self.d_model, 64)
        self.fc2 = nn.Linear(64, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
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
    def __init__(self, input_dim, seq_len=60, patch_len=12, stride=12, d_model=128, n_heads=4, e_layers=3, dropout=0.2):
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

    def forward(self, x):
        x = self.revin(x, 'norm')
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.reshape(patches.shape[0], patches.shape[1], -1)
        x = self.value_embedding(patches) + self.position_embedding
        x = self.encoder(x)
        x = self.head(x)
        return x

# ==============================================================================
# Helper Functions
# ==============================================================================
def get_halving_features(dates):
    dates = pd.to_datetime(dates)
    h_dates = pd.to_datetime(['2009-01-03', '2012-11-28', '2016-07-09', '2020-05-11', '2024-04-19'])
    h_rewards = [50.0, 25.0, 12.5, 6.25, 3.125]
    
    rewards, days_since, progress = [], [], []
    for d in dates:
        past_idx = np.where(h_dates <= d)[0]
        epoch_idx = 0 if len(past_idx) == 0 else past_idx[-1]
        
        reward = h_rewards[epoch_idx]
        last_h = h_dates[epoch_idx]
        diff_days = (d - last_h).days
        days_since.append(float(diff_days))
        rewards.append(reward)
        
        if epoch_idx < len(h_dates) - 1:
            next_h = h_dates[epoch_idx + 1]
            total_epoch_days = (next_h - last_h).days
            prog = diff_days / total_epoch_days
        else:
            next_h = pd.to_datetime('2028-04-17') 
            total_epoch_days = (next_h - last_h).days
            prog = diff_days / total_epoch_days
        progress.append(prog)
    return rewards, days_since, progress

def fetch_oos_data(start_date):
    """Fetch data from yfinance since the end of the training data."""
    ticker = yf.Ticker("BTC-USD")
    lookback_start = pd.to_datetime(start_date) - pd.Timedelta(days=65)
    df = ticker.history(start=lookback_start.strftime('%Y-%m-%d'))
    
    df.index = df.index.tz_localize(None)
    data = pd.DataFrame(index=df.index)
    data['Price'] = df['Close']
    data['Open'] = df['Open']
    data['High'] = df['High']
    data['Low'] = df['Low']
    data['Vol.'] = df['Volume']
    data['Change %'] = data['Price'].pct_change() * 100
    
    rewards, days_since, progress = get_halving_features(data.index)
    data['Block_Reward'] = rewards
    data['Days_Since_Halving'] = days_since
    data['Halving_Progress'] = progress
    
    data = data.dropna()
    return data

def run_forward_test(model_name, model_class, oos_data, scaler, seq_length=60):
    features = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %', 'Block_Reward', 'Days_Since_Halving', 'Halving_Progress']
    
    scaled_data = scaler.transform(oos_data[features])
    raw_prices = oos_data['Price'].values
    
    X, anchors, actuals, eval_dates = [], [], [], []
    
    for i in range(len(scaled_data) - seq_length - 7 + 1):
        X.append(scaled_data[i : i + seq_length])
        
        anchor_p = raw_prices[i + seq_length - 1]
        anchors.append(anchor_p)
        
        p_1d = raw_prices[i + seq_length]      # t+1
        p_3d = raw_prices[i + seq_length + 2]  # t+3
        p_7d = raw_prices[i + seq_length + 6]  # t+7
        
        actuals.append([p_1d, p_3d, p_7d])
        eval_dates.append(oos_data.index[i + seq_length - 1])
        
    X_t = torch.tensor(np.array(X), dtype=torch.float32)
    anchors = np.array(anchors)
    actuals = np.array(actuals)
    
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    if not os.path.exists(model_path):
        return None, None, None
        
    model = model_class(X_t.shape[2])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        y_pred_returns = model(X_t).numpy()
        
    pred_prices = np.zeros_like(actuals)
    pred_prices[:, 0] = anchors * np.exp(y_pred_returns[:, 0])
    pred_prices[:, 1] = anchors * np.exp(y_pred_returns[:, 1])
    pred_prices[:, 2] = anchors * np.exp(y_pred_returns[:, 2])
    
    return pred_prices, actuals, eval_dates

# ==============================================================================
# Streamlit UI
# ==============================================================================
st.markdown('<div class="cc-eyebrow">Evaluation</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-title">True OOS Forward Testing ⏱️</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-subtitle">Evaluate models on completely unseen market data from March 2024 to present.</div>', unsafe_allow_html=True)

if not os.path.exists(DATA_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Training data or scaler not found. Ensure previous steps are complete.")
    st.stop()

# Determine OOS gap
train_df = pd.read_csv(DATA_PATH)
last_train_date = pd.to_datetime(train_df['Date'].max())
today = pd.Timestamp.now().tz_localize(None)
gap_days = (today - last_train_date).days

c1, c2 = st.columns(2)
with c1:
    callout(
        "What is True Out-Of-Sample (OOS) Testing?",
        f"<p>The original dataset ended on <b>{last_train_date.strftime('%Y-%m-%d')}</b>. "
        f"Since today is <b>{today.strftime('%Y-%m-%d')}</b>, there are exactly <b>{gap_days} days</b> of new market data that the model has <i>never seen</i>.<br><br>"
        "Evaluating the models on this gap acts as a definitive <b>Forward Test</b> to prove real-world predictive robustness.</p>"
    )

with st.spinner(f"Fetching {gap_days} days of OOS data from Yahoo Finance..."):
    oos_df = fetch_oos_data(last_train_date.strftime('%Y-%m-%d'))
    
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)['scaler']

st.markdown('<div class="cc-section-title">Run Forward Test Pipeline</div>', unsafe_allow_html=True)

model_opt = st.selectbox("Select Model to Forward Test", ["LSTM", "Transformer", "PatchTST"])

if st.button("Execute OOS Forward Test", type="primary"):
    if model_opt == "LSTM":
        model_class = LSTMModel
    elif model_opt == "Transformer":
        model_class = TransformerModel
    else:
        model_class = PatchTSTModel
    
    with st.spinner(f"Running batch inference for {model_opt}..."):
        preds, actuals, eval_dates = run_forward_test(model_opt, model_class, oos_df, scaler)
        
    if preds is not None:
        st.session_state['ft_preds'] = preds
        st.session_state['ft_actuals'] = actuals
        st.session_state['ft_eval_dates'] = eval_dates
        st.session_state['ft_model'] = model_opt
    else:
        st.error(f"Saved weights for {model_opt} not found in models/ directory.")

if 'ft_preds' in st.session_state:
    preds = st.session_state['ft_preds']
    actuals = st.session_state['ft_actuals']
    eval_dates = st.session_state['ft_eval_dates']
    model_opt = st.session_state['ft_model']

    # Calculate Metrics
    metrics = []
    horizons = ['1D', '3D', '7D']
    for idx, h in enumerate(horizons):
        y_p = preds[:, idx]
        y_a = actuals[:, idx]
        
        mae = mean_absolute_error(y_a, y_p)
        rmse = np.sqrt(mean_squared_error(y_a, y_p))
        mape = mean_absolute_percentage_error(y_a, y_p) * 100
        
        metrics.append({
            "Horizon": h,
            "MAE (USD)": mae,
            "RMSE (USD)": rmse,
            "MAPE (%)": mape
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    st.markdown('<div class="cc-section-title">OOS Performance Metrics</div>', unsafe_allow_html=True)
    st.dataframe(
        metrics_df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "MAE (USD)": st.column_config.NumberColumn(format="$%.2f"),
            "RMSE (USD)": st.column_config.NumberColumn(format="$%.2f"),
            "MAPE (%)": st.column_config.NumberColumn(format="%.2f%%"),
        }
    )
    
    # Plot OOS Predictions
    st.markdown('<div class="cc-section-title">Out-of-Sample Predictions vs Actual Price</div>', unsafe_allow_html=True)
    
    horizon_opt = st.radio("Select Plot Horizon", ["1D", "3D", "7D"], horizontal=True)
    
    if horizon_opt == "1D":
        h_idx = 0
        plot_dates = [d + pd.Timedelta(days=1) for d in eval_dates]
    elif horizon_opt == "3D":
        h_idx = 1
        plot_dates = [d + pd.Timedelta(days=3) for d in eval_dates]
    else:
        h_idx = 2
        plot_dates = [d + pd.Timedelta(days=7) for d in eval_dates]
        
    fig = go.Figure()
    # Plot actuals and preds mapped to the exact evaluation dates to avoid lookback gap
    fig.add_trace(go.Scatter(x=plot_dates, y=actuals[:, h_idx], mode='lines', name=f'Actual {horizon_opt} Price', line=dict(color='#828b97', width=1)))
    fig.add_trace(go.Scatter(x=plot_dates, y=preds[:, h_idx], mode='lines', name=f'{model_opt} {horizon_opt} Prediction', line=dict(color='#29b57a', width=1.5, dash='dash')))
    
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, title="Date"),
        yaxis=dict(showgrid=True, gridcolor='#2b2b2b', title="Price (USD)"),
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
