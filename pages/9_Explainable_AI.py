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
    
    df = df.dropna()
    features = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %', 'Block_Reward', 'Days_Since_Halving', 'Halving_Progress']
    
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
    # Take a smaller background sample for speed
    background = _X_t[np.random.choice(_X_t.shape[0], 50, replace=False)]
    test_idx = np.random.choice(_X_t.shape[0], 10, replace=False)
    test_sample = _X_t[test_idx]
    
    try:
        # DeepExplainer can sometimes fail on complex ops like unfold.
        explainer = shap.DeepExplainer(_model, background)
        shap_vals = explainer.shap_values(test_sample)
    except Exception as e:
        # Fallback to GradientExplainer if DeepExplainer fails on PatchTST
        explainer = shap.GradientExplainer(_model, background)
        shap_vals = explainer.shap_values(test_sample)
    
    # Aggregate across the sequence dimension (dim 1)
    shap_1d = np.abs(shap_vals[0]).mean(axis=1).mean(axis=0)
    return shap_1d

# ==============================================================================
# Streamlit UI
# ==============================================================================
st.markdown('<div class="cc-eyebrow">Explainable AI</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-title">Model Explainability (SHAP) 🔮</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-subtitle">Demystify the neural network. Understand exactly which features drive predictions.</div>', unsafe_allow_html=True)

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

st.markdown(f'<div class="cc-section-title">Global Feature Importance ({model_opt})</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#8b949e;font-size:14px;margin-bottom:15px;">'
            'Using SHAP (SHapley Additive exPlanations) from cooperative game theory, we can assign an exact '
            'importance value to each feature. The chart below shows the average impact of each feature on '
            'the model\'s prediction of tomorrow\'s price across a random sample of the dataset.</p>', 
            unsafe_allow_html=True)

with st.spinner(f"Computing SHAP values for {model_opt} (this may take a few moments)..."):
    try:
        importance_scores = compute_shap_values(model, X_t, model_opt)
        
        sorted_idx = np.argsort(importance_scores)
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_scores = importance_scores[sorted_idx]
        
        fig = go.Figure(go.Bar(
            x=sorted_scores,
            y=sorted_features,
            orientation='h',
            marker=dict(
                color=sorted_scores,
                colorscale='Viridis',
                line=dict(color='rgba(0,0,0,0)', width=1)
            )
        ))
        
        fig.update_layout(
            **DARK_LAYOUT,
            title=dict(text=f"SHAP Feature Importance: {model_opt}", font=dict(color="#e6edf3")),
            xaxis_title="Mean |SHAP value| (Average Impact on Model Output)",
            yaxis_title="",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        top_feature = sorted_features[-1]
        st.info(f"💡 **Insight:** The SHAP explainer reveals that **{top_feature}** is the most significant driver for the {model_opt} model's predictions.")
        
    except Exception as e:
        st.error(f"Error computing SHAP values: {e}")
