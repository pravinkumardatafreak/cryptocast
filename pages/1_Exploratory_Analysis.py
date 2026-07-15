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
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.streamlit_utils import inject_custom_css, card, callout, DARK_LAYOUT
inject_custom_css()

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(PROJECT_DIR, "data", "btc_data.csv")
VIZ_DIR     = os.path.join(PROJECT_DIR, "visualizations")

# CSS Styles

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
            "Full Distribution",
            "Zoom to Center (-5% to +5%)",
            "Left Tail: Extreme Selloffs (-25% to -5%)",
            "Right Tail: Bullish Spikes (+5% to +25%)",
        ]
    )

    # Filter data based on zoom preset and show contextual info
    if "Center" in zoom_opt:
        st.info("**Normal Market Dynamics:** The vast majority of daily Bitcoin returns naturally fall within this -5% to +5% standard band.")
        plot_df = df_clean[(df_clean["Log_Return"] >= -5) & (df_clean["Log_Return"] <= 5)]
        range_x = [-6, 6]
        nbins = 80
    elif "Left" in zoom_opt:
        st.info("**Outliers & Black Swans:** Shows days with massive liquidity crises and sell-offs (e.g., COVID-19 crash, major exchange collapses).")
        plot_df = df_clean[(df_clean["Log_Return"] >= -25) & (df_clean["Log_Return"] <= -5)]
        range_x = [-26, -4]
        nbins = 40
    elif "Right" in zoom_opt:
        st.info("**Hyper-Bullish Spikes:** Captures extreme single-day rallies often driven by institutional adoption news or halving cycle supply shocks.")
        plot_df = df_clean[(df_clean["Log_Return"] >= 5) & (df_clean["Log_Return"] <= 25)]
        range_x = [4, 26]
        nbins = 40
    else:
        st.info("**Full Scope:** Visualizes the complete heavy-tailed, non-normal distribution of historical daily returns.")
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
        xaxis_range=range_x,
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
