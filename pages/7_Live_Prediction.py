import streamlit as st
import os
import sys
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import torch
import torch.nn as nn
from src.llm_insights import get_groq_api_key, generate_prediction_insight, generate_log_insights

# Set page config
st.set_page_config(
    page_title="CryptoCast | Live Prediction Demo",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.streamlit_utils import inject_custom_css, card, DARK_LAYOUT
inject_custom_css()

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
SCALER_PATH = os.path.join(PROJECT_DIR, 'scalers.pkl')

# ==============================================================================
# Model Architectures (Must exactly match train_model_pytorch.py)
# ==============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=128, batch_first=True)
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

def fetch_live_data():
    """Fetch recent data from yfinance and apply exact EDA preprocessing."""
    ticker = yf.Ticker("BTC-USD")
    # Fetch enough data to compute 60-day sequence + extra for pct_change
    df = ticker.history(period="100d")
    
    # OPTION A: Drop the current incomplete day to prevent distribution shift
    df = df.iloc[:-1]
    
    # Strip timezone from index to avoid alignment issues
    df.index = df.index.tz_localize(None)
    
    # Map yfinance columns to expected EDA columns
    data = pd.DataFrame(index=df.index)
    data['Price'] = df['Close']
    data['Open'] = df['Open']
    data['High'] = df['High']
    data['Low'] = df['Low']
    data['Vol.'] = df['Volume']
    # yfinance doesn't provide Change %, calculate it:
    data['Change %'] = data['Price'].pct_change() * 100
    
    # Add Whitepaper Halving features
    rewards, days_since, progress = get_halving_features(data.index)
    data['Block_Reward'] = rewards
    data['Days_Since_Halving'] = days_since
    data['Halving_Progress'] = progress
    
    data = data.dropna() # Drop NaN from pct_change
    return data

def run_inference(model_name, model_class, scaled_sequence, anchor_price):
    """Run inference for a specific PyTorch model."""
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    if not os.path.exists(model_path):
        return None
    
    input_dim = scaled_sequence.shape[-1]
    model = model_class(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Convert sequence to PyTorch tensor (batch_size=1)
    seq_tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        predicted_log_returns = model(seq_tensor).numpy()[0] # Shape: (3,)
        
    # Reconstruct prices from log returns: P_t+h = P_t * exp(r_h)
    predicted_prices = anchor_price * np.exp(predicted_log_returns)
    return predicted_prices

# ==============================================================================
# Streamlit UI
# ==============================================================================

st.markdown('<div class="cc-eyebrow">Real-Time Inference</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-title">Live Model Prediction Demo 🔴</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-subtitle">Fetch real-time Bitcoin data and run PyTorch forward passes instantly.</div>', unsafe_allow_html=True)

# 1. Fetch Live Data
with st.spinner("Fetching live market data from Yahoo Finance..."):
    live_df = fetch_live_data()

current_price = live_df['Price'].iloc[-1]
current_date = live_df.index[-1].strftime("%Y-%m-%d %H:%M")
price_change = live_df['Change %'].iloc[-1]

st.markdown('<div class="cc-section-title">Market State (Last Completed Close)</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Last Closed Date", current_date)
with col2:
    st.metric("Anchor Price (USD)", f"${current_price:,.2f}", f"{price_change:.2f}% (24h)")
with col3:
    st.metric("Input Sequence Length", "60 Completed Days")

# 2. Run Inference
st.markdown('<div class="cc-section-title">Model Inference</div>', unsafe_allow_html=True)

if not os.path.exists(SCALER_PATH):
    st.error("Scaler not found. Run `step1_eda.py` first.")
else:
    # Load Scaler
    with open(SCALER_PATH, 'rb') as f:
        scalers = pickle.load(f)
    scaler = scalers['scaler']
    
    # Prepare sequence
    features = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %', 'Block_Reward', 'Days_Since_Halving', 'Halving_Progress']
    # Get last 60 days of features
    last_60_days = live_df[features].iloc[-60:]
    scaled_sequence = scaler.transform(last_60_days)
    
    model_choice = st.selectbox("Select Model Architecture for Live Prediction:", ["LSTM", "Transformer", "PatchTST"])
    
    if st.button("Run Forward Pass", type="primary"):
        if model_choice == "LSTM":
            model_class = LSTMModel
        elif model_choice == "Transformer":
            model_class = TransformerModel
        else:
            model_class = PatchTSTModel
        
        with st.spinner(f"Running {model_choice} model inference..."):
            predicted_prices = run_inference(model_choice, model_class, scaled_sequence, current_price)
            
        if predicted_prices is None:
            st.error(f"Model weights for {model_choice} not found. Ensure you ran the training script to save `.pth` files.")
        else:
            st.session_state['live_pred_results'] = {
                'model_choice': model_choice,
                'predicted_prices': predicted_prices
            }
            p_1d, p_3d, p_7d = predicted_prices
            # Save to log
            log_path = os.path.join(PROJECT_DIR, 'data', 'live_predictions_log.csv')
            log_entry = pd.DataFrame([{
                'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Model': model_choice,
                'Current_Price': round(current_price, 2),
                'Pred_1D': round(p_1d, 2),
                'Pred_3D': round(p_3d, 2),
                'Pred_7D': round(p_7d, 2)
            }])
            if os.path.exists(log_path):
                log_entry.to_csv(log_path, mode='a', header=False, index=False)
            else:
                log_entry.to_csv(log_path, mode='w', header=True, index=False)
                
    if 'live_pred_results' in st.session_state:
        res = st.session_state['live_pred_results']
        model_choice = res['model_choice']
        p_1d, p_3d, p_7d = res['predicted_prices']
        
        # Display Predictions in Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            pct = ((p_1d - current_price) / current_price) * 100
            st.metric("1D Forecast (Today)", f"${p_1d:,.2f}", f"{pct:+.2f}%")
        with c2:
            pct = ((p_3d - current_price) / current_price) * 100
            st.metric("3D Forecast (Today +2D)", f"${p_3d:,.2f}", f"{pct:+.2f}%")
        with c3:
            pct = ((p_7d - current_price) / current_price) * 100
            st.metric("7D Forecast (Today +6D)", f"${p_7d:,.2f}", f"{pct:+.2f}%")
        
        # Plot the Trajectory
        # Historical 30 days + Predictions
        hist_plot_df = live_df.iloc[-30:]
        
        # Create Future Dates
        last_date = hist_plot_df.index[-1]
        future_dates = [last_date + pd.Timedelta(days=d) for d in [1, 3, 7]]
        
        fig = go.Figure()
        
        # Historical Line
        fig.add_trace(go.Scatter(
            x=hist_plot_df.index, y=hist_plot_df['Price'],
            mode='lines', name='Historical', line=dict(color='#c9d1d9', width=2)
        ))
        
        # Last point to anchor predictions visually
        pred_x = [last_date] + future_dates
        pred_y = [current_price, p_1d, p_3d, p_7d]
        
        fig.add_trace(go.Scatter(
            x=pred_x, y=pred_y,
            mode='lines+markers', name=f'{model_choice} Forecast',
            line=dict(color='#4ade80', width=3, dash='dot'),
            marker=dict(size=8, color='#4ade80')
        ))
        
        fig.update_layout(
            **DARK_LAYOUT,
            title="Live Forecast Trajectory",
            height=400,
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- AI Insights for Single Prediction ---
        st.markdown("---")
        api_key = get_groq_api_key()
    
        if st.button("🤖 Generate AI Insight", type="secondary"):
            if not api_key:
                st.warning("Please configure your Groq API Key in the sidebar to use AI Insights.")
            else:
                with st.spinner("Analyzing prediction with Groq..."):
                    pred_data = {
                        "Current BTC Price": f"${current_price:,.2f}",
                        "Model": model_choice,
                        "24h Change": f"{price_change:.2f}%",
                        "1D Forecast": f"${p_1d:,.2f} ({((p_1d - current_price) / current_price)*100:+.2f}%)",
                        "3D Forecast": f"${p_3d:,.2f} ({((p_3d - current_price) / current_price)*100:+.2f}%)",
                        "7D Forecast": f"${p_7d:,.2f} ({((p_7d - current_price) / current_price)*100:+.2f}%)"
                    }
                    insight = generate_prediction_insight(pred_data, api_key)
                    
                    st.markdown(
                        f'<div style="background:#1e293b; border:1px solid #334155; border-left:4px solid #3b82f6; '
                        f'border-radius:8px; padding:16px; margin-top:16px;">'
                        f'{insight}</div>',
                        unsafe_allow_html=True
                    )

# Display Prediction Log
st.markdown("---")
with st.expander("View and Manage Live Prediction Log"):
    log_path = os.path.join(PROJECT_DIR, 'data', 'live_predictions_log.csv')
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        
        # --- Delete Logs Feature ---
        if not log_df.empty:
            st.markdown("### Manage Logs")
            timestamps = log_df['Timestamp'].tolist()
            to_delete = st.multiselect("Select predictions to delete by Timestamp:", timestamps)
            if st.button("🗑️ Delete Selected Logs", type="primary") and to_delete:
                log_df = log_df[~log_df['Timestamp'].isin(to_delete)]
                log_df.to_csv(log_path, index=False)
                st.success("Deleted successfully!")
                st.rerun()
        
        # --- AI Insights for Log History ---
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📊 Analyze Prediction History"):
            api_key = get_groq_api_key()
            if not api_key:
                st.warning("Please configure your Groq API Key in the sidebar to use AI Insights.")
            else:
                with st.spinner("Analyzing prediction logs with Groq..."):
                    insight = generate_log_insights(log_df, api_key)
                    st.markdown(
                        f'<div style="background:#1e293b; border:1px solid #334155; border-left:4px solid #8b5cf6; '
                        f'border-radius:8px; padding:16px; margin-top:16px;">'
                        f'{insight}</div>',
                        unsafe_allow_html=True
                    )
    else:
        st.info("No predictions logged yet. Run a forward pass to generate your first log!")
