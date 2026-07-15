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
from src.streamlit_utils import inject_custom_css, card, callout, DARK_LAYOUT
inject_custom_css()

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# CSS Styles

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
