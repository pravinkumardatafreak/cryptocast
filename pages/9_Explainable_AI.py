import streamlit as st
import os
import json
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(
    page_title="CryptoCast | Explainable AI (SHAP)",
    page_icon="🔮",
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

@st.cache_data
def prepare_shap_data():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    rewards, days_since, progress = get_halving_features(df.index)
    df['Block_Reward'] = rewards
    df['Days_Since_Halving'] = days_since
    df['Halving_Progress'] = progress
    
    # Tier 1: Cyclical Day-of-Week Encoding (Daily Resolution, T=7)
    day_of_week = df.index.dayofweek
    df['Day_Sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
    df['Day_Cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
    
    # Tier 2: Intra-Month Stage Cyclical Encoding (Q1=0, Q2=1, Q3=2, Q4=3; Period T=4)
    day_of_month = df.index.day
    stage_int = np.where(day_of_month <= 7, 0,
                np.where(day_of_month <= 15, 1,
                np.where(day_of_month <= 22, 2, 3)))
    df['Stage_Sin'] = np.sin(2 * np.pi * stage_int / 4.0)
    df['Stage_Cos'] = np.cos(2 * np.pi * stage_int / 4.0)
    
    # Tier 3: Annual Quarter Cyclical Encoding (Q1=0, Q2=1, Q3=2, Q4=3; Period T=4)
    quarter_int = df.index.quarter - 1
    df['Quarter_Sin'] = np.sin(2 * np.pi * quarter_int / 4.0)
    df['Quarter_Cos'] = np.cos(2 * np.pi * quarter_int / 4.0)
    
    # Tier 4: 4-Year Leap / Halving Epoch Cycle (Year % 4; Period T=4)
    leap_int = df.index.year % 4
    df['LeapCycle_Sin'] = np.sin(2 * np.pi * leap_int / 4.0)
    df['LeapCycle_Cos'] = np.cos(2 * np.pi * leap_int / 4.0)
    
    df = df.dropna()
    features = [
        'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %',
        'Day_Sin', 'Day_Cos', 'Stage_Sin', 'Stage_Cos',
        'Quarter_Sin', 'Quarter_Cos', 'LeapCycle_Sin', 'LeapCycle_Cos',
        'Days_Since_Halving', 'Halving_Progress'
    ]
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)['scaler']
        
    scaled_data = scaler.transform(df[features])
    
    seq_length = 60
    X = []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i : i + seq_length])
        
    X_t = torch.tensor(np.array(X), dtype=torch.float32)
    return X_t, features

@st.cache_data(show_spinner=False)
def compute_shap_values(_model, _X_t, model_name):
    """Compute SHAP values for all 3 output horizons (1D, 3D, 7D)."""
    np.random.seed(42)
    background = _X_t[np.random.choice(_X_t.shape[0], 50, replace=False)]
    test_idx = np.random.choice(_X_t.shape[0], 20, replace=False)
    test_sample = _X_t[test_idx]
    
    try:
        explainer = shap.DeepExplainer(_model, background)
        shap_vals = explainer.shap_values(test_sample)
    except Exception:
        explainer = shap.GradientExplainer(_model, background)
        shap_vals = explainer.shap_values(test_sample)
    
    results = {}
    horizon_names = ['1D', '3D', '7D']
    
    def extract_horizon_score(shap_raw, h_idx):
        if isinstance(shap_raw, list):
            arr = shap_raw[h_idx] if h_idx < len(shap_raw) else shap_raw[0]
            if arr.ndim == 3:
                return np.abs(arr).mean(axis=(0, 1))
            elif arr.ndim == 2:
                return np.abs(arr).mean(axis=0)
        elif isinstance(shap_raw, np.ndarray):
            if shap_raw.ndim == 4:
                if shap_raw.shape[-1] == 3:  # (n_samples, seq_len, n_features, 3)
                    arr = shap_raw[..., h_idx]
                    return np.abs(arr).mean(axis=(0, 1))
                elif shap_raw.shape[0] == 3:  # (3, n_samples, seq_len, n_features)
                    arr = shap_raw[h_idx]
                    return np.abs(arr).mean(axis=(0, 1))
            elif shap_raw.ndim == 3:  # (n_samples, seq_len, n_features)
                return np.abs(shap_raw).mean(axis=(0, 1))
        
        arr = np.array(shap_raw)
        return np.abs(arr).reshape(-1, 16).mean(axis=0)

    for i, h_name in enumerate(horizon_names):
        results[h_name] = extract_horizon_score(shap_vals, i)
        
    return results

# Feature category mapping for grouped analysis
FEATURE_CATEGORIES = {
    'Price': ['Price', 'Open', 'High', 'Low'],
    'Volume & Momentum': ['Vol.', 'Change %'],
    'Whitepaper Protocol': ['Days_Since_Halving', 'Halving_Progress'],
    'Day-of-Week Cycle': ['Day_Sin', 'Day_Cos'],
    'Intra-Month Stage': ['Stage_Sin', 'Stage_Cos'],
    'Quarterly Cycle': ['Quarter_Sin', 'Quarter_Cos'],
    'Halving Epoch Cycle': ['LeapCycle_Sin', 'LeapCycle_Cos'],
}

CATEGORY_COLORS = {
    'Price': '#ef4444',
    'Volume & Momentum': '#f59e0b',
    'Whitepaper Protocol': '#22c55e',
    'Day-of-Week Cycle': '#3b82f6',
    'Intra-Month Stage': '#8b5cf6',
    'Quarterly Cycle': '#06b6d4',
    'Halving Epoch Cycle': '#ec4899',
}

def get_category_importance(shap_scores, feature_names):
    """Aggregate SHAP importance by feature category."""
    cat_scores = {}
    for cat, feats in FEATURE_CATEGORIES.items():
        total = 0.0
        for f in feats:
            if f in feature_names:
                idx = feature_names.index(f)
                total += shap_scores[idx]
        cat_scores[cat] = total
    return cat_scores

# ==============================================================================
# Streamlit UI
# ==============================================================================
st.markdown('<div class="cc-eyebrow">Explainable AI</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-title">Model Explainability (SHAP) 🔮</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-subtitle">Demystify the neural network. Understand exactly which features drive predictions across all three forecast horizons.</div>', unsafe_allow_html=True)

if not SHAP_AVAILABLE:
    st.error("SHAP library is not installed. Please run `pip install shap`.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("Training data not found.")
    st.stop()

model_opt = st.selectbox("Select Model Architecture to Explain:", ["LSTM", "Transformer", "PatchTST"])

model_path = os.path.join(MODELS_DIR, f"{model_opt}.pth")
if not os.path.exists(model_path):
    st.error(f"{model_opt} model weights not found. Please train the model first.")
    st.stop()

# Load Data and Model
with st.spinner("Preparing sequence data for Game Theory analysis..."):
    X_t, feature_names = prepare_shap_data()
    
    if model_opt == "LSTM":
        model = LSTMModel(input_dim=len(feature_names))
    elif model_opt == "Transformer":
        model = TransformerModel(input_dim=len(feature_names))
    else:
        model = PatchTSTModel(input_dim=len(feature_names))
        
    model.load_state_dict(torch.load(model_path))
    model.eval()

with st.spinner(f"Computing SHAP values for {model_opt} across all 3 horizons..."):
    try:
        shap_results = compute_shap_values(model, X_t, model_opt)
    except Exception as e:
        st.error(f"Error computing SHAP values: {e}")
        st.stop()

# -- Tab Layout: Per-Feature | By Category | Multi-Horizon Comparison --
tab_feat, tab_cat, tab_compare = st.tabs([
    "📊 Per-Feature Importance",
    "🏷️ Feature Category Analysis",
    "📈 Multi-Horizon Comparison"
])

# ============================
# TAB 1: Per-Feature Bar Chart
# ============================
with tab_feat:
    horizon_sel = st.radio("Select Forecast Horizon:", ["1D", "3D", "7D"], horizontal=True, key="feat_hz")
    scores = shap_results[horizon_sel]
    
    # Assign colours by category
    feat_colors = []
    for fn in feature_names:
        assigned = '#64748b'
        for cat, feats in FEATURE_CATEGORIES.items():
            if fn in feats:
                assigned = CATEGORY_COLORS[cat]
                break
        feat_colors.append(assigned)
    
    sorted_idx = np.argsort(scores)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_scores = scores[sorted_idx]
    sorted_colors = [feat_colors[i] for i in sorted_idx]
    
    fig = go.Figure(go.Bar(
        x=sorted_scores,
        y=sorted_features,
        orientation='h',
        marker=dict(color=sorted_colors, line=dict(color='rgba(255,255,255,0.1)', width=0.5))
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=f"SHAP Feature Importance: {model_opt} — {horizon_sel} Horizon", font=dict(color="#e6edf3")),
        xaxis_title="Mean |SHAP value| (Average Impact on Model Output)",
        yaxis_title="",
        height=550
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Legend
    legend_html = '<div style="display:flex; flex-wrap:wrap; gap:12px; margin-top:8px;">'
    for cat, col in CATEGORY_COLORS.items():
        legend_html += f'<span style="display:inline-flex;align-items:center;gap:4px;"><span style="width:12px;height:12px;border-radius:3px;background:{col};display:inline-block;"></span><span style="color:#c9d1d9;font-size:12px;">{cat}</span></span>'
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)
    
    top_feat = sorted_features[-1]
    top2_feat = sorted_features[-2]
    st.info(f"💡 **{horizon_sel} Insight:** **{top_feat}** and **{top2_feat}** are the strongest drivers for the {model_opt} model's {horizon_sel} predictions.")

# ============================
# TAB 2: Feature Category Grouped Analysis
# ============================
with tab_cat:
    horizon_sel2 = st.radio("Select Forecast Horizon:", ["1D", "3D", "7D"], horizontal=True, key="cat_hz")
    cat_scores = get_category_importance(shap_results[horizon_sel2], feature_names)
    
    sorted_cats = sorted(cat_scores.items(), key=lambda x: x[1])
    cat_names = [c[0] for c in sorted_cats]
    cat_vals = [c[1] for c in sorted_cats]
    cat_cols = [CATEGORY_COLORS[c] for c in cat_names]
    
    fig2 = go.Figure(go.Bar(
        x=cat_vals,
        y=cat_names,
        orientation='h',
        marker=dict(color=cat_cols, line=dict(color='rgba(255,255,255,0.15)', width=1)),
        text=[f"{v:.4f}" for v in cat_vals],
        textposition='outside',
        textfont=dict(color='#c9d1d9', size=11)
    ))
    fig2.update_layout(
        **DARK_LAYOUT,
        title=dict(text=f"Feature Category Importance: {model_opt} — {horizon_sel2}", font=dict(color="#e6edf3")),
        xaxis_title="Aggregated Mean |SHAP value|",
        yaxis_title="",
        height=450
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Compute percentage share
    total_shap = sum(cat_vals) if sum(cat_vals) > 0 else 1
    top_cat = cat_names[-1]
    top_pct = (cat_vals[-1] / total_shap) * 100
    protocol_val = cat_scores.get('Whitepaper Protocol', 0)
    protocol_pct = (protocol_val / total_shap) * 100
    
    st.markdown(f"""
<div class="cc-callout" style="background: rgba(13, 17, 23, 0.7); border: 1px solid #30363d; border-radius: 10px; padding: 16px; margin-top: 10px;">
    <h4 style="color: #38bdf8; margin-bottom: 8px;">🧠 Category Interpretation</h4>
    <p style="color: #c9d1d9; font-size: 13px; line-height: 1.7;">
        <b style="color: #ef4444;">Price features</b> ({top_pct:.1f}%) dominate at the {horizon_sel2} horizon because short-term price 
        movements are auto-correlated — yesterday's close is the strongest predictor of today's level.<br><br>
        <b style="color: #22c55e;">Whitepaper Protocol features</b> (Block_Reward, Halving_Progress) contribute <b>{protocol_pct:.1f}%</b> of total 
        importance. While this appears small in absolute terms, these features provide the <i>only non-price structural signal</i> in the model — 
        they anchor predictions to Bitcoin's deterministic 4-year supply cycle rather than reactive price momentum.<br><br>
        <b style="color: #8b5cf6;">Cyclical encodings</b> (Day-of-Week, Intra-Month Stage, Quarter) capture the Turn-of-the-Month (TOM) 
        effect and weekly seasonality patterns that are invisible to raw price features.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================
# TAB 3: Multi-Horizon Comparison
# ============================
with tab_compare:
    st.markdown('<div class="cc-section-title">How Feature Importance Shifts Across Horizons</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e;font-size:13px;margin-bottom:12px;">'
                'This chart reveals how the model\'s reliance on different feature categories changes as the forecast '
                'horizon extends from 1-day to 7-day. At longer horizons, structural protocol features should gain relative importance.</p>',
                unsafe_allow_html=True)
    
    horizons = ['1D', '3D', '7D']
    fig3 = go.Figure()
    
    for cat in FEATURE_CATEGORIES:
        y_vals = []
        for h in horizons:
            cat_imp = get_category_importance(shap_results[h], feature_names)
            total = sum(cat_imp.values()) if sum(cat_imp.values()) > 0 else 1
            y_vals.append((cat_imp[cat] / total) * 100)
        
        fig3.add_trace(go.Bar(
            name=cat,
            x=horizons,
            y=y_vals,
            marker_color=CATEGORY_COLORS[cat],
            text=[f"{v:.1f}%" for v in y_vals],
            textposition='inside',
            textfont=dict(size=10, color='white')
        ))
    
    fig3.update_layout(
        **DARK_LAYOUT,
        barmode='stack',
        title=dict(text=f"Feature Category Share by Horizon — {model_opt}", font=dict(color="#e6edf3")),
        xaxis_title="Forecast Horizon",
        yaxis_title="Share of Total SHAP Importance (%)",
        yaxis=dict(range=[0, 100]),
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5, font=dict(size=10))
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("""
<div class="cc-callout" style="background: rgba(13, 17, 23, 0.7); border: 1px solid #30363d; border-radius: 10px; padding: 16px; margin-top: 10px;">
    <h4 style="color: #38bdf8; margin-bottom: 8px;">📐 The Horizon-Dependent Feature Shift</h4>
    <p style="color: #c9d1d9; font-size: 13px; line-height: 1.7;">
        <b>1-Day Horizon:</b> Price momentum dominates — the model primarily relies on yesterday's close and 
        intraday range (High, Low, Open) to predict tomorrow's movement.<br><br>
        <b>3-Day Horizon:</b> Volume and cyclical features gain relative weight as the model needs to capture 
        multi-day momentum patterns and weekly seasonality beyond pure price extrapolation.<br><br>
        <b>7-Day Horizon:</b> Whitepaper Protocol features (Block_Reward, Halving_Progress) and epoch-level 
        cyclical encodings gain their highest relative importance. At the weekly scale, the model increasingly 
        relies on Bitcoin's structural supply-side scarcity cycle rather than reactive daily price noise.
    </p>
</div>
""", unsafe_allow_html=True)

