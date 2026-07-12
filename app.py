"""
CryptoCast - Bitcoin Price Forecasting Dashboard
=================================================
Interactive multi-page dashboard for the CryptoCast capstone project.
This main landing page provides an overview of the forecasting scope.
"""

import os
import streamlit as st

# Set page config (expanded sidebar is standard for navigation)
st.set_page_config(
    page_title="CryptoCast | Overview",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# CSS Styles
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background-color: #0d1117 !important;
            font-family: 'Inter', -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
        }
        [data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer { visibility: hidden; }
        .block-container { padding: 2rem 2.5rem; max-width: 1280px; }
        [data-testid="stSidebar"] {
            background-color: #161b22 !important;
            border-right: 1px solid #21262d;
        }
        [data-testid="stSidebar"] * { color: #c9d1d9 !important; }
        p, li, span, label { color: #c9d1d9; }
        h1, h2, h3, h4, h5, h6 { color: #e6edf3; }
        .cc-eyebrow {
            font-size: 11px; font-weight: 600; letter-spacing: 0.1em;
            text-transform: uppercase; color: #4ade80; margin-bottom: 6px;
        }
        .cc-title {
            font-size: 36px; font-weight: 700; color: #e6edf3;
            margin-bottom: 4px; letter-spacing: -0.02em; line-height: 1.2;
        }
        .cc-subtitle { font-size: 16px; color: #8b949e; margin-bottom: 28px; }
        .cc-section-title {
            font-size: 18px; font-weight: 600; color: #e6edf3;
            margin-top: 24px; margin-bottom: 12px;
            padding-bottom: 8px; border-bottom: 1px solid #21262d;
        }
        .cc-card {
            background: #161b22; border: 1px solid #21262d;
            border-radius: 10px; padding: 18px 20px; height: 100%;
        }
        .cc-card h4 { margin-top: 0; margin-bottom: 8px; font-size: 13px; color: #8b949e; font-weight: 500; }
        .cc-card .cc-value { margin: 0; font-size: 24px; font-weight: 700; color: #e6edf3; }
        .cc-card .cc-detail { margin-top: 6px; margin-bottom: 0; font-size: 11px; color: #8b949e; }
        .cc-callout {
            background: #161b22; border-left: 4px solid #4ade80;
            border-radius: 0 8px 8px 0; padding: 16px 20px; margin: 16px 0;
            border-top: 1px solid #21262d; border-right: 1px solid #21262d; border-bottom: 1px solid #21262d;
        }
        .cc-callout h4 { margin-top: 0; margin-bottom: 8px; font-size: 14px; font-weight: 600; color: #e6edf3; }
        .cc-callout p, .cc-callout li { margin: 0; font-size: 13px; color: #c9d1d9; line-height: 1.6; }
        .cc-tag {
            display: inline-block; background: #161b22; border: 1px solid #21262d;
            color: #8b949e; padding: 6px 12px; border-radius: 20px;
            font-size: 12px; font-weight: 500; margin-right: 8px; margin-bottom: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def card(label, value, detail=""):
    st.markdown(
        f'<div class="cc-card"><h4>{label}</h4>'
        f'<p class="cc-value">{value}</p>'
        f'<p class="cc-detail">{detail}</p></div>',
        unsafe_allow_html=True,
    )

def callout(title, body_html):
    st.markdown(
        f'<div class="cc-callout"><h4>{title}</h4>{body_html}</div>',
        unsafe_allow_html=True,
    )

# Sidebar indicator
with st.sidebar:
    st.markdown("### CryptoCast Navigation")
    st.caption("Select a page above to drill down into metrics, backtests, or market cycles.")

# Header
st.markdown('<div class="cc-eyebrow">Capstone Project</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-title">CryptoCast: Bitcoin Price Forecasting</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cc-subtitle">Comparing deep learning architectures for '
    '1-day, 3-day, and 7-day BTC price forecasts</div>',
    unsafe_allow_html=True,
)

# KPI card strip
c1, c2, c3, c4 = st.columns(4)
with c1:
    card("Dataset Size", "4,964 records", "Daily BTC price data, 2010-2024")
with c2:
    card("Sequence Length", "60 days", "Historical lookback window")
with c3:
    card("Forecast Horizons", "1D / 3D / 7D", "Multi-output target returns")
with c4:
    card("Backtesting", "3-Fold WFV", "Expanding window cross-validation")

# Navigation helper callout
callout(
    "📌 Interactive Sidebar Navigation (Power BI Style)",
    "Use the left sidebar navigation panel to browse page-by-page. "
    "Explore exploratory analysis, outlier risks, seasonality heatmaps, performance metrics, "
    "expanding window backtests, actual-vs-predicted diagnostics, and macroeconomic/halving dynamics."
)

st.markdown('<div class="cc-section-title">Problem Statement</div>', unsafe_allow_html=True)
st.write(
    "Bitcoin prices are highly volatile and shaped by complex, non-linear temporal dynamics. "
    "This project builds and compares deep learning architectures that learn from historical "
    "price sequences to forecast BTC prices across three horizons - supporting use cases such as "
    "short-term trading signals, multi-horizon algorithmic decision-making, and volatility-aware "
    "risk management."
)

st.markdown('<div class="cc-section-title">Model Architectures</div>', unsafe_allow_html=True)
a1, a2 = st.columns(2)
with a1:
    callout("1D-CNN",
        "<p>Convolutional filters extract local short-term patterns. "
        "Fast to train, strong on short horizons.</p>")
    callout("RNN",
        "<p>Baseline sequential model. Efficient, but limited by vanishing gradients over "
        "long sequences. Fast performer on the 1D and 3D horizons.</p>")
with a2:
    callout("LSTM",
        "<p>Gated memory cells retain long-range dependencies, reducing the vanishing gradient "
        "problem seen in vanilla RNNs. Best performer on 1D and 3D horizons.</p>")
    callout("Transformer",
        "<p>Self-attention captures global context across the full 60-day sequence. "
        "Best performer on the 7D horizon.</p>")

st.markdown('<div class="cc-section-title">Business Relevance</div>', unsafe_allow_html=True)
st.markdown(
    '<span class="cc-tag">Trading signal generation</span>'
    '<span class="cc-tag">Algorithmic multi-horizon decisions</span>'
    '<span class="cc-tag">Volatility &amp; risk management</span>'
    '<span class="cc-tag">Investment analytics dashboards</span>',
    unsafe_allow_html=True,
)
