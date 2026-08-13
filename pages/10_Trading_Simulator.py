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
        'Day_Sin', 'Day_Cos', 'Stage_Sin', 'Stage_Cos',
        'Quarter_Sin', 'Quarter_Cos', 'LeapCycle_Sin', 'LeapCycle_Cos',
        'Days_Since_Halving', 'Halving_Progress',
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


def run_hybrid_confluence_simulation(
    oos_data, threshold_z=1.0, lookback=20, risk_off_alloc=0.5, sma_period=50, initial_capital=10000.0
):
    """CryptoCast Dual-Directional Confluence Trading Strategy.

    Execution Rules:
      1. Top 3 AI Models (LSTM, Transformer, PatchTST) forecast 1D, 3D, and 7D horizons.
      2. BUY ENTRY (LONG): Top 3 models predict HIGHER prices across 1D/3D/7D, BUT actual price
         is moving in the opposite direction on Monday/Tuesday (trading BELOW weekly open).
      3. SELL ENTRY (SHORT): Top 3 models predict LOWER prices across 1D/3D/7D, BUT actual price
         is moving in the opposite direction on Monday/Tuesday (trading ABOVE weekly open).
      4. Macro 50-day SMA Overlay dynamically scales capital allocation (Risk-On vs Risk-Off).
    """
    prep = _prepare_all_models_data(oos_data)
    if prep is None:
        return None
    anchors, actuals, eval_dates, model_preds, buy_hold_btc = prep

    top3_names = [m for m in ["LSTM", "Transformer", "PatchTST"] if m in model_preds]
    if not top3_names:
        return None

    # 1. Macro 50-day SMA
    price_series = oos_data['Price']
    sma_series = price_series.rolling(window=sma_period, min_periods=1).mean()
    eval_dates_dt = pd.to_datetime(eval_dates)
    sma_eval = sma_series.reindex(eval_dates_dt).ffill().bfill().values

    # 2. Weekly Open Price Map
    week_opens_map = {}
    for idx_dt, row in oos_data.iterrows():
        monday_dt = idx_dt - pd.Timedelta(days=idx_dt.dayofweek)
        match = oos_data[oos_data.index >= monday_dt]
        if not match.empty:
            week_opens_map[idx_dt] = match["Open"].iloc[0]
        else:
            week_opens_map[idx_dt] = row["Open"]

    week_opens_eval = [week_opens_map.get(dt, anchors[i]) for i, dt in enumerate(eval_dates_dt)]

    cash = initial_capital
    position = 0  # +1 = LONG, -1 = SHORT, 0 = CASH
    btc_held = 0.0
    short_units = 0.0
    entry_price = 0.0

    equity_curve = []
    buy_hold_curve = []
    trades_executed = 0
    winning_trades = 0

    risk_on_days = 0
    risk_off_days = 0
    trade_log = []

    for t in range(len(eval_dates)):
        current_dt = eval_dates_dt[t]
        current_price = anchors[t]
        sma_val = sma_eval[t]
        week_open_price = week_opens_eval[t]

        is_risk_on = current_price >= sma_val
        if is_risk_on:
            risk_on_days += 1
        else:
            risk_off_days += 1

        # Current Portfolio Valuation
        if position == 1:
            portfolio_val = cash + (btc_held * current_price)
        elif position == -1:
            portfolio_val = cash + (entry_price - current_price) * short_units
        else:
            portfolio_val = cash

        equity_curve.append(portfolio_val)
        buy_hold_curve.append(buy_hold_btc * current_price)

        if t < lookback:
            continue

        # ── 1. Top 3 AI Model Consensus Across 1D, 3D, 7D ──
        all_preds_t = [model_preds[m][t] for m in top3_names]  # list of [1D, 3D, 7D]
        avg_pred_1d = np.mean([p[0] for p in all_preds_t])
        avg_pred_3d = np.mean([p[1] for p in all_preds_t])
        avg_pred_7d = np.mean([p[2] for p in all_preds_t])

        models_bullish_all = (avg_pred_1d > 0) and (avg_pred_3d > 0) and (avg_pred_7d > 0)
        models_bearish_all = (avg_pred_1d < 0) and (avg_pred_3d < 0) and (avg_pred_7d < 0)

        # ── 2. Monday/Tuesday Counter-Trend Market Conditions ──
        is_mon_or_tue = current_dt.dayofweek in [0, 1]
        below_week_open = current_price < week_open_price  # Price going down on Mon/Tue (Discount)
        above_week_open = current_price > week_open_price  # Price going up on Mon/Tue (Premium)

        # ── 3. Strategy Signal Rules ──
        buy_trigger = is_mon_or_tue and below_week_open and models_bullish_all
        sell_trigger = is_mon_or_tue and above_week_open and models_bearish_all

        # Sunday Weekly Reset
        is_sunday = current_dt.dayofweek == 6

        # ── EXITS ──
        if position == 1 and (sell_trigger or is_sunday or (avg_pred_7d < 0)):
            # Exit LONG
            cash = portfolio_val
            is_win = current_price > entry_price
            if is_win:
                winning_trades += 1
            trade_log.append({
                'date': current_dt.strftime('%Y-%m-%d'),
                'type': 'CLOSE LONG',
                'price': f"${current_price:,.2f}",
                'result': 'WIN 🟢' if is_win else 'LOSS 🔴',
                'pnl': f"{((current_price - entry_price) / entry_price * 100):+.2f}%"
            })
            position = 0
            btc_held = 0.0

        elif position == -1 and (buy_trigger or is_sunday or (avg_pred_7d > 0)):
            # Exit SHORT
            cash = portfolio_val
            is_win = current_price < entry_price
            if is_win:
                winning_trades += 1
            trade_log.append({
                'date': current_dt.strftime('%Y-%m-%d'),
                'type': 'CLOSE SHORT',
                'price': f"${current_price:,.2f}",
                'result': 'WIN 🟢' if is_win else 'LOSS 🔴',
                'pnl': f"{((entry_price - current_price) / entry_price * 100):+.2f}%"
            })
            position = 0
            short_units = 0.0

        # ── ENTRIES ──
        if buy_trigger and position == 0:
            alloc_ratio = 1.0 if is_risk_on else risk_off_alloc
            trade_cash = portfolio_val * alloc_ratio
            btc_held = trade_cash / current_price
            cash = portfolio_val - trade_cash
            entry_price = current_price
            position = 1
            trades_executed += 1
            trade_log.append({
                'date': current_dt.strftime('%Y-%m-%d'),
                'type': 'BUY (LONG)',
                'price': f"${current_price:,.2f}",
                'result': 'OPEN',
                'pnl': f"Mon/Tue Discount (${current_price:,.2f} < ${week_open_price:,.2f})"
            })

        elif sell_trigger and position == 0:
            alloc_ratio = 1.0 if is_risk_on else risk_off_alloc
            trade_cash = portfolio_val * alloc_ratio
            short_units = trade_cash / current_price
            cash = portfolio_val
            entry_price = current_price
            position = -1
            trades_executed += 1
            trade_log.append({
                'date': current_dt.strftime('%Y-%m-%d'),
                'type': 'SELL (SHORT)',
                'price': f"${current_price:,.2f}",
                'result': 'OPEN',
                'pnl': f"Mon/Tue Premium (${current_price:,.2f} > ${week_open_price:,.2f})"
            })

    final_price = actuals[-1]
    if position == 1:
        final_portfolio_val = cash + (btc_held * final_price)
    elif position == -1:
        final_portfolio_val = cash + (entry_price - final_price) * short_units
    else:
        final_portfolio_val = cash

    equity_curve.append(final_portfolio_val)
    buy_hold_curve.append(buy_hold_btc * final_price)
    plot_dates = list(eval_dates) + [eval_dates[-1] + pd.Timedelta(days=1)]

    # Calculate Max Drawdowns
    eq_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(eq_arr)
    strat_dd = (eq_arr - peak) / peak * 100
    max_strat_dd = np.min(strat_dd)

    bh_arr = np.array(buy_hold_curve)
    bh_peak = np.maximum.accumulate(bh_arr)
    bh_dd = (bh_arr - bh_peak) / bh_peak * 100
    max_bh_dd = np.min(bh_dd)

    total_eval_days = len(eval_dates)
    risk_on_pct = (risk_on_days / total_eval_days * 100) if total_eval_days > 0 else 0.0
    risk_off_pct = (risk_off_days / total_eval_days * 100) if total_eval_days > 0 else 0.0

    # Sharpe Ratio
    daily_returns = np.diff(eq_arr) / eq_arr[:-1]
    sharpe = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-10)) * np.sqrt(365)

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
        'max_strat_dd': max_strat_dd,
        'max_bh_dd': max_bh_dd,
        'risk_on_pct': risk_on_pct,
        'risk_off_pct': risk_off_pct,
        'sharpe': sharpe,
        'trade_log': trade_log,
        'top3_models': top3_names,
    }

from src.streamlit_utils import render_stakeholder_narrative

# ==============================================================================
# Streamlit UI
# ==============================================================================
render_stakeholder_narrative(
    page_num=10,
    total_pages=11,
    title="Trading Bot Simulator",
    simple_explanation="This page translates deep learning prediction accuracy into real-world Business ROI, capital growth, and risk-managed portfolio execution.",
    connection_story="Connects model predictions (Pages 3, 7, 8) and statistical hypothesis tests (Page 11) to simulate algorithmic execution on live out-of-sample market data.",
    key_takeaway="Combines Top 3 AI Model multi-horizon predictions (1D/3D/7D) with Monday/Tuesday counter-trend entries (buying discounts below weekly open, selling premiums above weekly open)."
)

st.markdown('<div class="cc-eyebrow">Financial Impact</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-title">Trading Bot Simulator 💸</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-subtitle">Dual-Directional (BUY & SELL) AI Strategy — Top 3 Models (1D/3D/7D) + Monday/Tuesday Weekly Open Counter-Trend Signals.</div>', unsafe_allow_html=True)

if not os.path.exists(DATA_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Training data or scaler not found. Ensure previous steps are complete.")
    st.stop()

train_df = pd.read_csv(DATA_PATH)
last_train_date = pd.to_datetime(train_df['Date'].max())

# ── Strategy Architecture Explanation ──────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(13,17,23,0.9), rgba(30,41,59,0.9)); border:1px solid #30363d; border-radius:12px; padding:20px; margin-bottom:20px;">
    <h4 style="color:#38bdf8; margin-bottom:12px;">🧬 Dual-Directional (BUY & SELL) Strategy Architecture</h4>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
        <div style="background:rgba(34,197,94,0.1); border:1px solid rgba(34,197,94,0.3); border-radius:8px; padding:12px;">
            <span style="color:#22c55e; font-weight:700;">🟢 BUY ENTRY (LONG)</span>
            <p style="color:#94a3b8; font-size:12px; margin:4px 0 0;">
                <b>Model Signal:</b> Top 3 models (LSTM, Transformer, PatchTST) ALL predict higher prices across 1D, 3D, and 7D.<br>
                <b>Market Trigger:</b> On Mon/Tue, price is moving down <i>below weekly open</i> (discount entry).
            </p>
        </div>
        <div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); border-radius:8px; padding:12px;">
            <span style="color:#ef4444; font-weight:700;">🔴 SELL ENTRY (SHORT)</span>
            <p style="color:#94a3b8; font-size:12px; margin:4px 0 0;">
                <b>Model Signal:</b> Top 3 models ALL predict lower prices across 1D, 3D, and 7D.<br>
                <b>Market Trigger:</b> On Mon/Tue, price is moving up <i>above weekly open</i> (premium exit/short).
            </p>
        </div>
    </div>
    <p style="color:#e2e8f0; font-size:13px; margin-top:14px; text-align:center; font-weight:600;">
        🛡️ Macro 50-Day SMA Risk Overlay: Full capital allocation in Risk-On (Price ≥ SMA), reduced in Risk-Off (Price < SMA)
    </p>
</div>
""", unsafe_allow_html=True)

# ── Tunable Parameters ─────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    threshold_val = st.number_input(
        "Z-Score Threshold (σ)",
        min_value=0.0, max_value=3.0, value=1.0, step=0.1,
        help="Layer 4 fires when price Z-score drops below this negative threshold.",
    )
with col2:
    lookback_days = st.slider(
        "Lookback Window (days)",
        min_value=5, max_value=60, value=20, step=5,
        help="Rolling window for Z-score computation.",
    )
with col3:
    risk_off_alloc_val = st.slider(
        "Risk-Off Capital (%)",
        min_value=10, max_value=90, value=50, step=10,
        help="Capital allocation when Price < 50-day SMA (Risk-Off regime).",
    )

if st.button("🚀 Run Simulation on Unseen Data", type="primary", use_container_width=True):
    with st.spinner("Fetching unseen out-of-sample data from Yahoo Finance..."):
        import time
        ticker = yf.Ticker("BTC-USD")
        lookback_start = pd.to_datetime(last_train_date) - pd.Timedelta(days=65)
        oos_df = None
        for attempt in range(1, 4):
            try:
                oos_df = fetch_oos_data(last_train_date.strftime('%Y-%m-%d'))
                if oos_df is not None and not oos_df.empty:
                    break
            except Exception:
                pass
            time.sleep(attempt * 2)

        if oos_df is None or oos_df.empty:
            st.error("⚠️ Yahoo Finance API temporarily unavailable. Please retry in 30 seconds.")
            st.stop()

    with st.spinner("Running CryptoCast Confluence Strategy..."):
        results = run_hybrid_confluence_simulation(
            oos_df,
            threshold_z=threshold_val,
            lookback=lookback_days,
            risk_off_alloc=risk_off_alloc_val / 100.0,
        )

        st.session_state['sim_results'] = results

if 'sim_results' in st.session_state and st.session_state['sim_results'] is not None:
    results = st.session_state['sim_results']

    st.markdown(
        '<div class="cc-section-title">CryptoCast Confluence Strategy — Out-Of-Sample Results</div>',
        unsafe_allow_html=True,
    )

    ai_roi = ((results['final_val'] - 10000) / 10000) * 100
    bh_roi = ((results['buy_hold_val'] - 10000) / 10000) * 100

    ai_color = "negative" if ai_roi < 0 else ""
    sharpe_color = "negative" if results.get('sharpe', 0) < 0 else ""

    # ── 6-Column Metric Dashboard ──────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.markdown(f'<div class="metric-card"><div class="metric-value {ai_color}">${results["final_val"]:,.0f}</div><div class="metric-label">Strategy Value</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="metric-value {ai_color}">{ai_roi:+.2f}%</div><div class="metric-label">Strategy ROI</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><div class="metric-value">{results["trades_executed"]}</div><div class="metric-label">Total Trades</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><div class="metric-value">{results["win_rate"]:.1f}%</div><div class="metric-label">Win Rate</div></div>', unsafe_allow_html=True)
    m5.markdown(f'<div class="metric-card"><div class="metric-value">{results.get("max_strat_dd", 0.0):.1f}%</div><div class="metric-label">Max Drawdown</div></div>', unsafe_allow_html=True)
    m6.markdown(f'<div class="metric-card"><div class="metric-value {sharpe_color}">{results.get("sharpe", 0.0):.2f}</div><div class="metric-label">Sharpe Ratio</div></div>', unsafe_allow_html=True)

    # ── Benchmark Comparison Row ───────────────────────────────────────────────
    bh_color = "negative" if bh_roi < 0 else ""
    b1, b2, b3 = st.columns([1, 1, 2])
    b1.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#94a3b8;">${results["buy_hold_val"]:,.0f}</div><div class="metric-label">Buy & Hold Value</div></div>', unsafe_allow_html=True)
    b2.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#94a3b8;">{bh_roi:+.2f}%</div><div class="metric-label">Buy & Hold ROI</div></div>', unsafe_allow_html=True)
    alpha = ai_roi - bh_roi
    alpha_color = "#4ade80" if alpha > 0 else "#f87171"
    b3.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{alpha_color};">{alpha:+.2f}%</div><div class="metric-label">Alpha Generated vs Benchmark</div></div>', unsafe_allow_html=True)

    if 'risk_on_pct' in results:
        st.markdown(
            f'<div style="background:#0d1117; border:1px solid #30363d; border-radius:8px; '
            f'padding:12px 16px; margin-top:16px; margin-bottom:8px; font-size:13px; color:#c9d1d9;">'
            f'🌐 <b>Macro Liquidity Regime</b>: '
            f'<span style="color:#4ade80; font-weight:700;">Risk-On</span>: {results["risk_on_pct"]:.1f}% of period | '
            f'<span style="color:#fb923c; font-weight:700;">Risk-Off</span>: {results["risk_off_pct"]:.1f}% of period'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Interactive Subplot: Equity Curve + Drawdown Fill ──────────────────────
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Portfolio Equity Growth ($10,000 Initial Capital)", "Portfolio Drawdown (%)")
    )

    # Top: Equity Curve
    fig.add_trace(go.Scatter(
        x=results['dates'], y=results['equity_curve'],
        mode='lines', name='CryptoCast Confluence Strategy',
        line=dict(color='#4ade80', width=2.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=results['dates'], y=results['buy_hold_curve'],
        mode='lines', name='Bitcoin Buy & Hold Benchmark',
        line=dict(color='#828b97', width=1.5, dash='dot')
    ), row=1, col=1)

    # Bottom: Drawdown Fill
    strat_equity = np.array(results['equity_curve'])
    strat_peaks = np.maximum.accumulate(strat_equity)
    strat_dd = (strat_equity - strat_peaks) / strat_peaks * 100

    bh_equity = np.array(results['buy_hold_curve'])
    bh_peaks = np.maximum.accumulate(bh_equity)
    bh_dd = (bh_equity - bh_peaks) / bh_peaks * 100

    fig.add_trace(go.Scatter(
        x=results['dates'], y=strat_dd,
        mode='lines', name='Strategy Drawdown',
        fill='tozeroy', fillcolor='rgba(248, 113, 113, 0.25)',
        line=dict(color='#f87171', width=1.5)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=results['dates'], y=bh_dd,
        mode='lines', name='Benchmark Drawdown',
        line=dict(color='#828b97', width=1, dash='dash')
    ), row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=520,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Value (USD)", tickformat="$,", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ── Visual Row 2: Win-Rate Donut Chart & Risk Regime Pie ──────────────────
    c_pie1, c_pie2 = st.columns(2)

    with c_pie1:
        win_rate = results.get("win_rate", 50.0)
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Winning Trades', 'Losing Trades'],
            values=[win_rate, 100.0 - win_rate],
            hole=.6,
            marker_colors=['#4ade80', '#f87171'],
            textinfo='label+percent'
        )])
        fig_donut.update_layout(
            **DARK_LAYOUT,
            title="Signal Win-Loss Distribution",
            height=280,
            showlegend=False
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_pie2:
        risk_on_pct = results.get("risk_on_pct", 70.0)
        risk_off_pct = results.get("risk_off_pct", 30.0)
        fig_regime = go.Figure(data=[go.Pie(
            labels=['Risk-On (100% Capital)', 'Risk-Off (Capital Preserved)'],
            values=[risk_on_pct, risk_off_pct],
            hole=.6,
            marker_colors=['#38bdf8', '#fb923c'],
            textinfo='label+percent'
        )])
        fig_regime.update_layout(
            **DARK_LAYOUT,
            title="Macro Capital Allocation Distribution",
            height=280,
            showlegend=False
        )
        st.plotly_chart(fig_regime, use_container_width=True)

    if ai_roi > bh_roi:
        st.success(
            f"**Alpha Generated!** The CryptoCast Confluence Strategy "
            f"outperformed the market benchmark by "
            f"**{ai_roi - bh_roi:+.2f}%** in the unseen period."
        )
    else:
        st.warning(
            f"The Confluence Strategy underperformed the benchmark by {bh_roi - ai_roi:.2f}%. "
            f"This is expected during sustained bull runs where Buy & Hold captures full upside. "
            f"The strategy's value is in **drawdown protection** — check Max Drawdown comparison above."
        )

    # ── Trade Log (Expandable) ────────────────────────────────────────────────
    trade_log = results.get('trade_log', [])
    if trade_log:
        with st.expander(f"📋 Trade Execution Log ({len(trade_log)} entries)", expanded=False):
            log_df = pd.DataFrame(trade_log)
            st.dataframe(log_df, use_container_width=True, hide_index=True)

    # --- AI Insights for Trading Simulation ---
    st.markdown("---")
    api_key = get_groq_api_key()

    if st.button("🤖 Explain Strategy Performance", use_container_width=True):
        if not api_key:
            st.warning("Please configure your Groq API Key in the sidebar to use AI Insights.")
        else:
            with st.spinner("Analyzing simulation results with Groq..."):
                sim_results_for_llm = results.copy()
                sim_results_for_llm.pop('trade_log', None)
                sim_results_for_llm['strategy_name'] = 'CryptoCast Confluence Strategy'
                sim_results_for_llm['models'] = 'PatchTST (Champion)'

                insight = generate_trading_insight(sim_results_for_llm, api_key)
                st.markdown(
                    f'<div style="background:#1e293b; border:1px solid #334155; border-left:4px solid #10b981; '
                    f'border-radius:8px; padding:16px; margin-top:16px;">'
                    f'{insight}</div>',
                    unsafe_allow_html=True
                )

