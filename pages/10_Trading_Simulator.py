import streamlit as st
import os
import json
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import yfinance as yf
from src.llm_insights import get_groq_api_key, generate_trading_insight

st.set_page_config(
    page_title="CryptoCast | Trading Simulator",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI
st.markdown("""
<style>
    .cc-title { font-size: 2rem; font-weight: 700; color: #e6edf3; margin-bottom: 0.2rem; }
    .cc-subtitle { font-size: 1.1rem; color: #8b949e; margin-bottom: 2rem; }
    .cc-section-title { font-size: 1.3rem; font-weight: 600; color: #e6edf3; margin-top: 2rem; margin-bottom: 1rem; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }
    .cc-eyebrow { font-size: 0.85rem; font-weight: 600; color: #38bdf8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: -15px;}
    .metric-card { background-color: rgba(22,27,34,0.6); border: 1px solid #30363d; border-radius: 8px; padding: 15px; text-align: center; }
    .metric-value { font-size: 24px; font-weight: 700; color: #4ade80; }
    .metric-value.negative { color: #f87171; }
    .metric-label { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'btc_data.csv')
SCALER_PATH = os.path.join(PROJECT_DIR, 'scalers.pkl')

DARK_LAYOUT = dict(
    template='plotly_dark',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=40, r=40, t=40, b=40)
)

# ==============================================================================
# Model Architectures
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
# Helpers
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

# Historical 1D MAE from model_comparison_results.csv — used to rank
# the top-2 models for the ensemble confluence signal.
_MODEL_1D_MAE = {
    "LSTM":        755.38,
    "Transformer": 986.47,
    "PatchTST":   1050.82,  # Closest available proxy (RNN-class)
}


def _build_sequences(oos_data):
    """Scale features and build 60-day sliding-window sequences.

    Returns:
        tuple: (X_t, anchors, actuals, eval_dates, buy_hold_btc,
                input_dim)  or None on failure.
    """
    features = [
        'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %',
        'Block_Reward', 'Days_Since_Halving', 'Halving_Progress',
    ]

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)['scaler']

    scaled_data = scaler.transform(oos_data[features])
    raw_prices = oos_data['Price'].values

    seq_length = 60
    X, anchors, actuals, eval_dates = [], [], [], []

    for i in range(len(scaled_data) - seq_length - 1):
        X.append(scaled_data[i : i + seq_length])
        anchors.append(raw_prices[i + seq_length - 1])
        actuals.append(raw_prices[i + seq_length])
        eval_dates.append(oos_data.index[i + seq_length - 1])

    X_t = torch.tensor(np.array(X), dtype=torch.float32)
    anchors = np.array(anchors)
    actuals = np.array(actuals)

    buy_hold_btc = 10000.0 / anchors[0]
    return X_t, anchors, actuals, eval_dates, buy_hold_btc, len(features)


def _load_and_infer(model_name, X_t, input_dim):
    """Load a single model's weights and return its predicted returns.

    Returns:
        np.ndarray of shape (N, 3) or None if weights are missing.
    """
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    if not os.path.exists(model_path):
        return None

    class_map = {
        "LSTM":        LSTMModel,
        "Transformer": TransformerModel,
        "PatchTST":    PatchTSTModel,
    }
    model = class_map[model_name](input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        return model(X_t).numpy()


def _prepare_simulation_data(model_opt, oos_data):
    """Prepare sequences and run inference for a SINGLE model.

    Returns:
        tuple: (anchors, actuals, eval_dates, y_pred_returns,
                buy_hold_btc) or None if model weights are missing.
    """
    seq = _build_sequences(oos_data)
    if seq is None:
        return None
    X_t, anchors, actuals, eval_dates, buy_hold_btc, input_dim = seq

    y_pred = _load_and_infer(model_opt, X_t, input_dim)
    if y_pred is None:
        return None

    return anchors, actuals, eval_dates, y_pred, buy_hold_btc


def _prepare_all_models_data(oos_data):
    """Run inference for ALL available models and return per-model
    predicted returns.

    Returns:
        tuple: (anchors, actuals, eval_dates, model_preds_dict,
                buy_hold_btc) where model_preds_dict maps model name
                → np.ndarray (N,3).  Returns None if no model has
                valid weights.
    """
    seq = _build_sequences(oos_data)
    if seq is None:
        return None
    X_t, anchors, actuals, eval_dates, buy_hold_btc, input_dim = seq

    model_preds = {}
    for name in ["LSTM", "Transformer", "PatchTST"]:
        preds = _load_and_infer(name, X_t, input_dim)
        if preds is not None:
            model_preds[name] = preds

    if not model_preds:
        return None

    return anchors, actuals, eval_dates, model_preds, buy_hold_btc


def run_momentum_simulation(
    model_opt, oos_data, threshold_pct, initial_capital=10000.0
):
    """Model-Momentum strategy.

    De-means the model's 1-day predicted log-return to remove constant
    bias, then treats the residual as a directional signal:
      * signal > threshold  →  BUY  (model is more bullish than usual)
      * signal < -threshold →  SELL (model is less bullish than usual)
    """
    prep = _prepare_simulation_data(model_opt, oos_data)
    if prep is None:
        return None
    anchors, actuals, eval_dates, y_pred_returns, buy_hold_btc = prep

    # De-mean the signal to remove constant model bias
    raw_signal = y_pred_returns[:, 0]
    signal_mean = np.mean(raw_signal)
    demeaned_signal = raw_signal - signal_mean

    # ── Portfolio simulation ──────────────────────────────────────────
    cash = initial_capital
    btc_held = 0.0
    equity_curve = []
    buy_hold_curve = []
    trades_executed = 0
    winning_trades = 0
    in_position = False
    threshold = threshold_pct / 100.0

    for t in range(len(eval_dates)):
        current_price = anchors[t]
        signal = demeaned_signal[t]

        portfolio_val = cash + (btc_held * current_price)
        equity_curve.append(portfolio_val)
        buy_hold_curve.append(buy_hold_btc * current_price)

        # --- Trading Rules ---
        if signal > threshold and not in_position:
            btc_held = cash / current_price
            cash = 0.0
            in_position = True
            trades_executed += 1
            if actuals[t] > current_price:
                winning_trades += 1

        elif signal < -threshold and in_position:
            cash = btc_held * current_price
            btc_held = 0.0
            in_position = False
            trades_executed += 1
            if actuals[t] < current_price:
                winning_trades += 1

    # End-of-simulation mark-to-market
    final_price = actuals[-1]
    final_portfolio_val = cash + (btc_held * final_price)
    equity_curve.append(final_portfolio_val)
    buy_hold_curve.append(buy_hold_btc * final_price)
    plot_dates = list(eval_dates) + [eval_dates[-1] + pd.Timedelta(days=1)]

    return {
        'dates': plot_dates,
        'equity_curve': equity_curve,
        'buy_hold_curve': buy_hold_curve,
        'trades_executed': trades_executed,
        'win_rate': (
            (winning_trades / trades_executed * 100)
            if trades_executed > 0 else 0.0
        ),
        'final_val': final_portfolio_val,
        'buy_hold_val': buy_hold_curve[-1],
    }


def run_mean_reversion_simulation(
    oos_data, threshold_z, lookback=20,
    initial_capital=10000.0,
):
    """Mean-Reversion + Top-2 Ensemble Confluence strategy.

    This is a *hybrid* strategy that only trades when TWO independent
    signals agree on the same direction:

    Signal 1 — **Rolling Z-Score (statistical)**
        Measures how far the current price deviates from its N-day
        rolling mean in standard-deviation units.  A z-score below
        −threshold means the price is unusually low; above +threshold
        means unusually high.

    Signal 2 — **Top-2 Model Ensemble Direction (learned)**
        Runs inference on ALL three models (LSTM, Transformer,
        PatchTST), ranks them by historical 1D MAE, and averages
        the predicted 1-day log-return of the best two.  If this
        average is positive the ensemble is *bullish*; if negative,
        *bearish*.

    Trading rules (confluence — both must agree):
        BUY  when  z_score < −threshold  AND  ensemble is bullish
        SELL when  z_score >  threshold  AND  ensemble is bearish

    Args:
        oos_data:        Out-of-sample DataFrame with price features.
        threshold_z:     Z-score threshold (in standard deviations).
        lookback:        Rolling window for mean / std (days).
        initial_capital: Starting portfolio value in USD.

    Returns:
        dict with equity curves, trade stats, and top-2 model names,
        or None if fewer than 2 models have valid weights.
    """
    prep = _prepare_all_models_data(oos_data)
    if prep is None:
        return None
    anchors, actuals, eval_dates, model_preds, buy_hold_btc = prep

    # ── Rank models by 1D MAE and pick top 2 ─────────────────────────
    available = sorted(
        model_preds.keys(),
        key=lambda m: _MODEL_1D_MAE.get(m, 9999),
    )
    if len(available) < 2:
        return None  # Need at least 2 models for ensemble

    top2_names = available[:2]

    # Average the 1-day predicted log-return of the top 2 models
    ensemble_signal = np.mean(
        [model_preds[m][:, 0] for m in top2_names], axis=0,
    )
    # De-mean to remove constant bias
    ensemble_signal = ensemble_signal - np.mean(ensemble_signal)

    # ── Portfolio simulation ──────────────────────────────────────────
    cash = initial_capital
    btc_held = 0.0
    equity_curve = []
    buy_hold_curve = []
    trades_executed = 0
    winning_trades = 0
    in_position = False

    for t in range(len(eval_dates)):
        current_price = anchors[t]

        portfolio_val = cash + (btc_held * current_price)
        equity_curve.append(portfolio_val)
        buy_hold_curve.append(buy_hold_btc * current_price)

        # Need enough history for the rolling window
        if t < lookback:
            continue

        # ── Signal 1: Rolling z-score ─────────────────────────────────
        window_prices = anchors[t - lookback : t]
        rolling_mean = np.mean(window_prices)
        rolling_std = np.std(window_prices)
        if rolling_std < 1e-8:
            continue

        z_score = (current_price - rolling_mean) / rolling_std

        # ── Signal 2: Top-2 ensemble direction ────────────────────────
        ensemble_bullish = ensemble_signal[t] > 0
        ensemble_bearish = ensemble_signal[t] < 0

        # ── Confluence: both signals must agree ───────────────────────
        if (z_score < -threshold_z
                and ensemble_bullish
                and not in_position):
            # Price below mean + top-2 models bullish → BUY
            btc_held = cash / current_price
            cash = 0.0
            in_position = True
            trades_executed += 1
            if actuals[t] > current_price:
                winning_trades += 1

        elif (z_score > threshold_z
                and ensemble_bearish
                and in_position):
            # Price above mean + top-2 models bearish → SELL
            cash = btc_held * current_price
            btc_held = 0.0
            in_position = False
            trades_executed += 1
            if actuals[t] < current_price:
                winning_trades += 1

    # End-of-simulation mark-to-market
    final_price = actuals[-1]
    final_portfolio_val = cash + (btc_held * final_price)
    equity_curve.append(final_portfolio_val)
    buy_hold_curve.append(buy_hold_btc * final_price)
    plot_dates = list(eval_dates) + [eval_dates[-1] + pd.Timedelta(days=1)]

    return {
        'dates': plot_dates,
        'equity_curve': equity_curve,
        'buy_hold_curve': buy_hold_curve,
        'trades_executed': trades_executed,
        'win_rate': (
            (winning_trades / trades_executed * 100)
            if trades_executed > 0 else 0.0
        ),
        'final_val': final_portfolio_val,
        'buy_hold_val': buy_hold_curve[-1],
        'top2_models': top2_names,
    }

# ==============================================================================
# Streamlit UI
# ==============================================================================
st.markdown('<div class="cc-eyebrow">Financial Impact</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-title">Trading Bot Simulator 💸</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-subtitle">Translate mathematical accuracy into real-world Business ROI by simulating a trading strategy.</div>', unsafe_allow_html=True)

if not os.path.exists(DATA_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Training data or scaler not found. Ensure previous steps are complete.")
    st.stop()

train_df = pd.read_csv(DATA_PATH)
last_train_date = pd.to_datetime(train_df['Date'].max())

# ── Strategy Descriptions ─────────────────────────────────────────────────────
STRATEGY_INFO = {
    "Model Momentum": (
        "**Trend-Following** — Buys when the model is *more* bullish than its "
        "own historical average and sells when it turns relatively bearish. "
        "Works best in trending markets."
    ),
    "Mean Reversion": (
        "**Confluence Strategy** — Trades ONLY when two independent signals "
        "agree: (1) a rolling z-score detects that price is significantly "
        "above or below its mean, AND (2) the average predicted direction "
        "of the **top-2 models** (ranked by 1D MAE: LSTM + Transformer) "
        "confirms the same direction. This dual-filter approach reduces "
        "false signals and works well in range-bound markets."
    ),
}

# Strategy selector first — other controls depend on this choice
strategy_opt = st.selectbox(
    "Trading Strategy:",
    list(STRATEGY_INFO.keys()),
    help="Choose the trading logic that drives buy/sell decisions.",
)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if strategy_opt == "Mean Reversion":
        st.markdown(
            '<div style="padding-top:28px; color:#4ade80; font-size:13px; '
            'font-weight:600;">🤖 Uses Top-2 Ensemble<br>'
            '(LSTM + Transformer)</div>',
            unsafe_allow_html=True,
        )
        model_opt = None  # Not needed — ensemble uses all models
    else:
        model_opt = st.selectbox(
            "Select AI Brain:", ["LSTM", "Transformer", "PatchTST"],
        )
with col2:
    if strategy_opt == "Mean Reversion":
        threshold_val = st.number_input(
            "Z-Score Threshold (σ)",
            min_value=0.0, max_value=3.0, value=1.0, step=0.1,
            help="Buy/sell when price deviates this many standard deviations from its rolling mean.",
        )
    else:
        threshold_val = st.number_input(
            "Trade Threshold (%)",
            min_value=0.0, max_value=5.0, value=0.0, step=0.1,
            help="Only trade if expected return exceeds this % (to cover fees).",
        )
with col3:
    if strategy_opt == "Mean Reversion":
        lookback_days = st.slider(
            "Lookback Window (days)",
            min_value=5, max_value=60, value=20, step=5,
            help="Number of past days used to calculate the rolling mean and std deviation.",
        )
    else:
        st.markdown(
            '<div style="padding-top:28px; color:#8b949e; font-size:13px;">'
            'No extra parameters for Momentum.</div>',
            unsafe_allow_html=True,
        )

# Show strategy explanation
st.markdown(
    f'<div style="background:#161b22; border:1px solid #30363d; border-radius:8px; '
    f'padding:12px 16px; margin-bottom:16px; font-size:14px; color:#c9d1d9;">'
    f'📖 {STRATEGY_INFO[strategy_opt]}</div>',
    unsafe_allow_html=True,
)

if st.button("Run Simulation on Unseen Data", type="primary"):
    with st.spinner("Fetching unseen OOS data from Yahoo Finance..."):
        oos_df = fetch_oos_data(last_train_date.strftime('%Y-%m-%d'))
        
    with st.spinner(f"Running {strategy_opt} strategy..."):
        if strategy_opt == "Mean Reversion":
            results = run_mean_reversion_simulation(
                oos_df,
                threshold_z=threshold_val,
                lookback=lookback_days,
            )
        else:
            results = run_momentum_simulation(
                model_opt, oos_df,
                threshold_pct=threshold_val,
            )
        
        st.session_state['sim_results'] = results
        st.session_state['sim_strategy'] = strategy_opt
        st.session_state['sim_model'] = None if strategy_opt == "Mean Reversion" else model_opt

if 'sim_results' in st.session_state and st.session_state['sim_results'] is not None:
    results = st.session_state['sim_results']
    saved_strategy = st.session_state['sim_strategy']
    saved_model = st.session_state['sim_model']
    
    # Build a label showing which models are in play
    if saved_strategy == "Mean Reversion":
        top2 = results.get('top2_models', [])
        model_label = f"Top-2 Ensemble ({' + '.join(top2)})"
    else:
        model_label = saved_model

        st.markdown(
            f'<div class="cc-section-title">{strategy_opt} Results · '
            f'{model_label} (Out-Of-Sample Data)</div>',
            unsafe_allow_html=True,
        )
        
        ai_roi = ((results['final_val'] - 10000) / 10000) * 100
        bh_roi = ((results['buy_hold_val'] - 10000) / 10000) * 100
        
        ai_color = "negative" if ai_roi < 0 else ""
        
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-value {ai_color}">${results["final_val"]:,.2f}</div><div class="metric-label">{strategy_opt} Value</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-value {ai_color}">{ai_roi:+.2f}%</div><div class="metric-label">{strategy_opt} ROI</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-value">{results["trades_executed"]}</div><div class="metric-label">Total Trades Executed</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-value">{results["win_rate"]:.1f}%</div><div class="metric-label">Signal Win Rate</div></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['dates'], y=results['equity_curve'], 
            mode='lines', name=f'{model_label} {strategy_opt}', 
            line=dict(color='#4ade80', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=results['dates'], y=results['buy_hold_curve'], 
            mode='lines', name='Buy and Hold Benchmark', 
            line=dict(color='#828b97', width=1.5, dash='dot')
        ))
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title=dict(text="Portfolio Equity Curve ($10,000 Initial Capital)", font=dict(color="#e6edf3")),
            xaxis_title="Date",
            yaxis_title="Portfolio Value (USD)",
            yaxis=dict(tickformat="$,"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if ai_roi > bh_roi:
            st.success(
                f"**Alpha Generated!** The {model_label} {saved_strategy} "
                f"strategy outperformed the market benchmark by "
                f"**{ai_roi - bh_roi:+.2f}%** in the unseen period."
            )
        else:
            tip = (
                "Try adjusting the Z-Score Threshold or Lookback Window."
                if saved_strategy == "Mean Reversion"
                else "Try adjusting the Trade Threshold to filter out noise."
            )
            st.warning(
                f"The {model_label} {saved_strategy} strategy underperformed "
                f"the benchmark. {tip}"
            )
            
        # --- AI Insights for Trading Simulation ---
        st.markdown("---")
        api_key = get_groq_api_key()
        
        if st.button("🤖 Explain Strategy Performance", use_container_width=True):
            if not api_key:
                st.warning("Please configure your Groq API Key in the sidebar to use AI Insights.")
            else:
                with st.spinner("Analyzing simulation results with Groq..."):
                    # Add strategy metadata to results for the LLM
                    sim_results_for_llm = results.copy()
                    sim_results_for_llm['strategy_name'] = saved_strategy
                    sim_results_for_llm['models'] = model_label
                    
                    insight = generate_trading_insight(sim_results_for_llm, api_key)
                    st.markdown(
                        f'<div style="background:#1e293b; border:1px solid #334155; border-left:4px solid #10b981; '
                        f'border-radius:8px; padding:16px; margin-top:16px;">'
                        f'{insight}</div>',
                        unsafe_allow_html=True
                    )

