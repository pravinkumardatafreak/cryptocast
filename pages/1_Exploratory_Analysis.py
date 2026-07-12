import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Set page config
st.set_page_config(
    page_title="CryptoCast | Exploratory Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(PROJECT_DIR, "data", "btc_data.csv")
VIZ_DIR     = os.path.join(PROJECT_DIR, "visualizations")

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
            font-size: 32px; font-weight: 700; color: #e6edf3;
            margin-bottom: 4px; letter-spacing: -0.02em; line-height: 1.2;
        }
        .cc-subtitle { font-size: 14px; color: #8b949e; margin-bottom: 28px; }
        .cc-section-title {
            font-size: 18px; font-weight: 600; color: #e6edf3;
            margin-top: 24px; margin-bottom: 12px;
            padding-bottom: 8px; border-bottom: 1px solid #21262d;
        }
        .cc-callout {
            background: #161b22; border-left: 4px solid #4ade80;
            border-radius: 0 8px 8px 0; padding: 16px 20px; margin: 16px 0;
            border-top: 1px solid #21262d; border-right: 1px solid #21262d; border-bottom: 1px solid #21262d;
        }
        .cc-callout.warn { border-left-color: #fb923c; }
        .cc-callout h4 { margin-top: 0; margin-bottom: 8px; font-size: 14px; font-weight: 600; color: #e6edf3; }
        .cc-callout p, .cc-callout li { margin: 0; font-size: 13px; color: #c9d1d9; line-height: 1.6; }
    </style>
    """,
    unsafe_allow_html=True,
)

DARK_LAYOUT = dict(
    plot_bgcolor="#0d1117",
    paper_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", color="#8b949e"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", color="#8b949e"),
    margin=dict(t=30, b=30, l=10, r=10),
)

def callout(title, body_html, warn=False):
    cls = "cc-callout warn" if warn else "cc-callout"
    st.markdown(
        f'<div class="{cls}"><h4>{title}</h4>{body_html}</div>',
        unsafe_allow_html=True,
    )

@st.cache_data(ttl=0)
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    return None

df_raw = load_data()

st.markdown('<div class="cc-eyebrow">Data exploration</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-subtitle">Audit historical Bitcoin trend, distributions, and outlier anomalies</div>', unsafe_allow_html=True)

if df_raw is not None:
    st.markdown('<div class="cc-section-title">Historical Price Trend</div>', unsafe_allow_html=True)
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=df_raw.index, y=df_raw["Price"],
        name="Close", line=dict(color="#4ade80", width=1.6)
    ))
    fig_price.add_trace(go.Scatter(
        x=df_raw.index, y=df_raw["Open"],
        name="Open", line=dict(color="#6e7681", width=0.9, dash="dot")
    ))
    fig_price.update_layout(
        **DARK_LAYOUT,
        xaxis_title="Date", yaxis_title="Price (USD)",
        hovermode="x unified", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9")),
    )
    st.plotly_chart(fig_price, use_container_width=True)

    with st.expander("View recent raw data"):
        st.dataframe(df_raw.tail(50), use_container_width=True)

    # ⚠️ Outlier Risk Section
    st.markdown('<div class="cc-section-title">⚠️ Outlier Risk & Metric Distortion</div>', unsafe_allow_html=True)
    st.write(
        "In financial time series forecasting, **outliers** (extreme price surges or flash crashes) "
        "present a critical challenge to model convergence and evaluation."
    )
    
    o1, o2 = st.columns(2)
    with o1:
        callout(
            "Why Outliers Skew Metrics (The Squaring Effect)",
            "<p>Standard metrics like **RMSE** (Root Mean Squared Error) square the errors before averaging them. "
            "This means a single prediction error on a crash day (like the March 2020 COVID crash) is weighted exponentially "
            "heavier than normal forecasting errors. If the model predicts $8,000 but the price drops to $4,800 (error of $3,200), "
            "the squared penalty is 10,240,000! This can falsely indicate that a model is performing poorly overall.</p>"
        )
        callout(
            "The Mitigation Strategy: Log Returns",
            "<p>To neutralize outlier risk, we do not train our networks on absolute prices. Instead, we train them on "
            "**stationary log returns**. Log returns compress extreme price shocks into scale-invariant changes, helping "
            "the gradient optimization stay stable and avoiding massive gradient explosions during backpropagation.</p>"
        )
    with o2:
        st.markdown("**Box Plot: The Gold Standard for Outlier Detection**")
        st.write(
            "A **Box-and-Whisker Plot** is the best visual tool for identifying outliers. "
            "The box represents the middle 50% of the data (Interquartile Range, IQR), the middle line shows the median, "
            "and any points plotted individually beyond the whiskers represent mathematical outliers."
        )
        box_img_path = os.path.join(VIZ_DIR, "06_seasonal_boxplots.png")
        if os.path.exists(box_img_path):
            st.image(box_img_path, caption="Box-and-Whisker Plot: Monthly Distribution & Outlier Points", use_container_width=True)

    st.markdown('<div class="cc-section-title">🔍 Interactive Return Distribution Explorer</div>', unsafe_allow_html=True)
    st.write(
        "Use this interactive histogram to zoom in and inspect the distribution of Bitcoin's daily log returns. "
        "Drag your mouse to zoom in on any section, or select a preset zoom from the dropdown below."
    )

    # Compute daily log returns
    df_raw["Log_Return"] = np.log(df_raw["Price"] / df_raw["Price"].shift(1)) * 100
    df_clean = df_raw.dropna()

    mean_ret = df_clean["Log_Return"].mean()
    std_ret  = df_clean["Log_Return"].std()
    min_ret  = df_clean["Log_Return"].min()
    max_ret  = df_clean["Log_Return"].max()
    min_date = df_clean["Log_Return"].idxmin().strftime("%Y-%m-%d")
    max_date = df_clean["Log_Return"].idxmax().strftime("%Y-%m-%d")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        card("Daily Return Mean", f"{mean_ret:.3f}%", "Positive structural drift")
    with m_col2:
        card("Daily Volatility (Std Dev)", f"{std_ret:.2f}%", "Standard dispersion width")
    with m_col3:
        card("Extreme Flash Crash", f"{min_ret:.1f}%", f"On {min_date} (COVID/Halving)")
    with m_col4:
        card("Extreme Bull Spike", f"{max_ret:.1f}%", f"On {max_date}")

    zoom_opt = st.selectbox(
        "Select Distribution Zoom Preset",
        [
            "Full Distribution (All Data)",
            "Zoom to Center (-5% to +5% - Normal Market Dynamics)",
            "Zoom to Left Tail (-25% to -5% - Extreme Selloffs & Outliers)",
            "Zoom to Right Tail (+5% to +25% - Hyper-Bullish Spikes)",
        ]
    )

    # Filter data based on zoom preset
    if "Center" in zoom_opt:
        plot_df = df_clean[(df_clean["Log_Return"] >= -5) & (df_clean["Log_Return"] <= 5)]
        range_x = [-6, 6]
        nbins = 80
    elif "Left" in zoom_opt:
        plot_df = df_clean[(df_clean["Log_Return"] >= -25) & (df_clean["Log_Return"] <= -5)]
        range_x = [-26, -4]
        nbins = 40
    elif "Right" in zoom_opt:
        plot_df = df_clean[(df_clean["Log_Return"] >= 5) & (df_clean["Log_Return"] <= 25)]
        range_x = [4, 26]
        nbins = 40
    else:
        plot_df = df_clean
        range_x = [df_clean["Log_Return"].min() - 2, df_clean["Log_Return"].max() + 2]
        nbins = 150

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=plot_df["Log_Return"],
        nbinsx=nbins,
        marker=dict(color="#38bdf8", line=dict(color="#0d1117", width=0.5)),
        hovertemplate="Return Bin: %{x:.2f}%<br>Count: %{y}<extra></extra>"
    ))

    # Add reference lines for full view or center view
    if "Full" in zoom_opt or "Center" in zoom_opt:
        # Mean line
        fig_dist.add_vline(x=mean_ret, line_dash="dash", line_color="#ffffff", line_width=1.5,
                           annotation_text="Mean", annotation_font_color="#ffffff", annotation_position="top left")
        # 1-Std Dev
        fig_dist.add_vline(x=mean_ret - std_ret, line_dash="dot", line_color="#fb923c", line_width=1.2,
                           annotation_text="-1σ", annotation_font_color="#fb923c", annotation_position="top left")
        fig_dist.add_vline(x=mean_ret + std_ret, line_dash="dot", line_color="#fb923c", line_width=1.2,
                           annotation_text="+1σ", annotation_font_color="#fb923c", annotation_position="top right")
        # 2-Std Dev (Threshold of outliers)
        fig_dist.add_vline(x=mean_ret - (2 * std_ret), line_dash="dash", line_color="#f87171", line_width=1.2,
                           annotation_text="-2σ (Outlier Threshold)", annotation_font_color="#f87171", annotation_position="top left")
        fig_dist.add_vline(x=mean_ret + (2 * std_ret), line_dash="dash", line_color="#f87171", line_width=1.2,
                           annotation_text="+2σ (Outlier Threshold)", annotation_font_color="#f87171", annotation_position="top right")

    fig_dist.update_layout(
        **DARK_LAYOUT,
        xaxis_title="Daily Log Return (%)",
        yaxis_title="Frequency (Days)",
        height=400,
        xaxis=dict(range=range_x, gridcolor="#21262d", zerolinecolor="#30363d", color="#8b949e"),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown('<div class="cc-section-title">Statistical Visualizations</div>', unsafe_allow_html=True)
    eda_images = {
        "Price Distribution":  "04_price_distribution.png",
        "Return Distribution": "05_return_distribution.png",
        "Seasonal Boxplots":   "06_seasonal_boxplots.png",
        "Correlation Heatmap": "07_correlation_heatmap.png",
        "Rolling Statistics":  "08_rolling_statistics.png",
    }
    selected_img = st.selectbox("Select visualization", list(eda_images.keys()))
    img_path = os.path.join(VIZ_DIR, eda_images[selected_img])
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.info("Visualization not found. Run step1_eda.py to generate it.")
else:
    st.error("data/btc_data.csv not found.")
