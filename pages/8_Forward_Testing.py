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

# Import model architectures from centralized src.models module
from src.models import CNN1D, RNNModel, LSTMModel, TransformerModel, RevIN, PatchTSTModel

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
    
    # Tier 1: Cyclical Day-of-Week Encoding (Daily Resolution, T=7)
    day_of_week = data.index.dayofweek
    data['Day_Sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
    data['Day_Cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
    
    # Tier 2: Intra-Month Stage Cyclical Encoding (Q1=0, Q2=1, Q3=2, Q4=3; Period T=4)
    day_of_month = data.index.day
    stage_int = np.where(day_of_month <= 7, 0,
                np.where(day_of_month <= 15, 1,
                np.where(day_of_month <= 22, 2, 3)))
    data['Stage_Sin'] = np.sin(2 * np.pi * stage_int / 4.0)
    data['Stage_Cos'] = np.cos(2 * np.pi * stage_int / 4.0)
    
    # Tier 3: Annual Quarter Cyclical Encoding (Q1=0, Q2=1, Q3=2, Q4=3; Period T=4)
    quarter_int = data.index.quarter - 1
    data['Quarter_Sin'] = np.sin(2 * np.pi * quarter_int / 4.0)
    data['Quarter_Cos'] = np.cos(2 * np.pi * quarter_int / 4.0)
    
    # Tier 4: 4-Year Leap / Halving Epoch Cycle (Year % 4; Period T=4)
    leap_int = data.index.year % 4
    data['LeapCycle_Sin'] = np.sin(2 * np.pi * leap_int / 4.0)
    data['LeapCycle_Cos'] = np.cos(2 * np.pi * leap_int / 4.0)
    
    data = data.dropna()
    return data

def run_forward_test(model_name, model_class, oos_data, scaler, seq_length=60):
    features = [
        'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %',
        'Day_Sin', 'Day_Cos', 'Stage_Sin', 'Stage_Cos',
        'Quarter_Sin', 'Quarter_Cos', 'LeapCycle_Sin', 'LeapCycle_Cos',
        'Days_Since_Halving', 'Halving_Progress'
    ]
    
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

    # Calculate Metrics & MDA
    from src.directional_bias import compute_mda
    # Extract anchor prices from raw oos data for exact MDA computation
    anchor_prices = oos_df['Price'].values[-len(preds):]
    mda_dict = compute_mda(actuals, preds, anchor_prices)

    metrics = []
    horizons = ['1D', '3D', '7D']
    for idx, h in enumerate(horizons):
        y_p = preds[:, idx]
        y_a = actuals[:, idx]
        
        mae = mean_absolute_error(y_a, y_p)
        rmse = np.sqrt(mean_squared_error(y_a, y_p))
        mape = mean_absolute_percentage_error(y_a, y_p) * 100
        mda_val = mda_dict.get(f"MDA_{h}", 0.0)
        
        metrics.append({
            "Horizon": h,
            "MAE (USD)": mae,
            "RMSE (USD)": rmse,
            "MAPE (%)": mape,
            "MDA Accuracy (%)": mda_val
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    st.markdown('<div class="cc-section-title">OOS Performance & Directional Accuracy Metrics</div>', unsafe_allow_html=True)
    st.dataframe(
        metrics_df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "MAE (USD)": st.column_config.NumberColumn(format="$%.2f"),
            "RMSE (USD)": st.column_config.NumberColumn(format="$%.2f"),
            "MAPE (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "MDA Accuracy (%)": st.column_config.NumberColumn(format="%.2f%%"),
        }
    )
    
    # Plot OOS Predictions & Alert Lines
    st.markdown('<div class="cc-section-title">Out-of-Sample Predictions vs Dynamic Alert Levels</div>', unsafe_allow_html=True)
    
    horizon_opt = st.radio("Select Plot Display Mode", ["1D", "3D", "7D", "Multi-Horizon Overlay (3 Alert Lines)"], horizontal=True)
    
    fig = go.Figure()
    plot_dates_anchor = [d for d in eval_dates]

    if horizon_opt == "Multi-Horizon Overlay (3 Alert Lines)":
        fig.add_trace(go.Scatter(x=plot_dates_anchor, y=actuals[:, 0], mode='lines', name='Actual BTC Close', line=dict(color='#e6edf3', width=2)))
        fig.add_trace(go.Scatter(x=plot_dates_anchor, y=preds[:, 0], mode='lines', name='1D Alert Line (Fast Momentum)', line=dict(color='#38bdf8', width=1.5, dash='solid')))
        fig.add_trace(go.Scatter(x=plot_dates_anchor, y=preds[:, 1], mode='lines', name='3D Alert Line (Swing Target)', line=dict(color='#c084fc', width=1.5, dash='dash')))
        fig.add_trace(go.Scatter(x=plot_dates_anchor, y=preds[:, 2], mode='lines', name='7D Alert Line (Macro Trend)', line=dict(color='#fbbf24', width=2, dash='dot')))
    else:
        if horizon_opt == "1D":
            h_idx = 0
            plot_dates = [d + pd.Timedelta(days=1) for d in eval_dates]
        elif horizon_opt == "3D":
            h_idx = 1
            plot_dates = [d + pd.Timedelta(days=3) for d in eval_dates]
        else:
            h_idx = 2
            plot_dates = [d + pd.Timedelta(days=7) for d in eval_dates]
            
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
