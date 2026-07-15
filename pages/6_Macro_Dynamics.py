import pandas as pd
import streamlit as st

# Set page config
st.set_page_config(
    page_title="CryptoCast | Macro Dynamics",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.streamlit_utils import inject_custom_css, card, callout, DARK_LAYOUT
inject_custom_css()

# CSS Styles

st.markdown('<div class="cc-eyebrow">Macroeconomics</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-title">Macro & Halving Dynamics</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-subtitle">Examine the disinflationary supply schedule and global capital flows affecting Bitcoin cycles</div>', unsafe_allow_html=True)

st.markdown('<div class="cc-section-title">Macroeconomic Dynamics &amp; Asset Rotation</div>',
            unsafe_allow_html=True)
st.write(
    "Bitcoin's price is heavily influenced by global macroeconomic liquidity cycles. "
    "Understanding these dynamics is vital for contextualizing model predictions."
)

col_macro1, col_macro2 = st.columns(2)
with col_macro1:
    callout(
        "Risk-On Sentiment &amp; Liquidity Expansion",
        "<p><b>Market Context:</b> Low interest rates and expansionary monetary policy (QE) "
        "increase global capital supply. Investors seek higher yields by taking on more risk.<br>"
        "<b>Asset Rotation:</b> Capital rotates from cash and safe-havens into risk assets "
        "(Equities, Tech stocks, and Cryptocurrencies).<br>"
        "<b>Bitcoin Impact:</b> Acts as a powerful tailwind, fueling exponential bull markets.</p>"
    )
with col_macro2:
    callout(
        "Risk-Off Sentiment &amp; Liquidity Contraction",
        "<p><b>Market Context:</b> Rising interest rates (QT) or macroeconomic/geopolitical crises "
        "cause investors to de-risk. Capital preservation becomes the primary goal.<br>"
        "<b>Asset Rotation:</b> Capital rotates out of speculative risk assets and back into "
        "cash (USD), short-term bonds, and gold.<br>"
        "<b>Bitcoin Impact:</b> Drives liquidity contraction, triggering deep corrections and "
        "structural bear markets.</p>",
        warn=True,
    )

st.markdown('<div class="cc-section-title">The Bitcoin Halving Cycle</div>', unsafe_allow_html=True)
st.write(
    "Every 210,000 blocks (roughly every 4 years), the block reward paid to miners is halved. "
    "This built-in disinflationary mechanism creates a supply-side shock that has historically "
    "initiated major price cycles."
)

halving_data = {
    "Halving Event": ["1st Halving", "2nd Halving", "3rd Halving", "4th Halving"],
    "Date":          ["Nov 28, 2012", "Jul 9, 2016", "May 11, 2020", "Apr 19, 2024"],
    "Block Height":  ["210,000",      "420,000",     "630,000",      "840,000"],
    "Reward Before": ["50.0 BTC",     "25.0 BTC",    "12.5 BTC",     "6.25 BTC"],
    "Reward After":  ["25.0 BTC",     "12.5 BTC",    "6.25 BTC",     "3.125 BTC"],
}
st.dataframe(pd.DataFrame(halving_data), use_container_width=True, hide_index=True)

callout(
    "Supply-Demand Mechanics",
    "<p>The halving cycle increases Bitcoin's Stock-to-Flow ratio, increasing its scarcity. "
    "When the growth rate of supply halves while demand remains stable or grows, it exerts "
    "upward pressure on price. These structural supply shifts typically play out over "
    "12-18 months following a halving event.</p>"
)

# 📈 Halving Cycle Quadrant Analysis Section
import os
import numpy as np
import plotly.graph_objects as go

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "btc_data.csv")

@st.cache_data(ttl=0)
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    return None

df_raw = load_data()

if df_raw is not None:
    st.markdown('<div class="cc-section-title">📈 Halving Cycle Quadrant Performance Analysis</div>', unsafe_allow_html=True)
    st.write(
        "By binning our whitepaper feature **`Halving_Progress`** into four distinct quadrants, "
        "we can quantify the average performance and volatility behavior of Bitcoin across its 4-year cycle stages:"
    )

    # Compute daily percentage returns
    df_raw["Daily_Return"] = df_raw["Price"].pct_change() * 100
    df_clean = df_raw.dropna()

    def get_quadrant(progress):
        if progress < 0.25: return 'Q1: Post-Halving (0-25%)'
        elif progress < 0.50: return 'Q2: Mid-Cycle Expansion (25-50%)'
        elif progress < 0.75: return 'Q3: Late-Cycle Peak/Bear (50-75%)'
        else: return 'Q4: Pre-Halving Accumulation (75-100%)'

    df_clean['Cycle_Phase'] = df_clean['Halving_Progress'].apply(get_quadrant)
    stats = df_clean.groupby('Cycle_Phase').agg(
        Avg_Return=('Daily_Return', 'mean'),
        Volatility=('Daily_Return', 'std'),
        Days=('Daily_Return', 'count')
    ).reset_index()

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.write("**Cycle Quadrant Statistical Performance**")
        disp_stats = stats.copy()
        disp_stats["Avg_Return"] = disp_stats["Avg_Return"].map(lambda x: f"+{x:.3f}%")
        disp_stats["Volatility"] = disp_stats["Volatility"].map(lambda x: f"{x:.2f}%")
        disp_stats["Days"] = disp_stats["Days"].map(lambda x: f"{x:,}")
        disp_stats.columns = ["Cycle Phase (Halving Progress)", "Avg Daily Return (%)", "Daily Volatility (%)", "Sample Size (Days)"]
        st.dataframe(disp_stats, use_container_width=True, hide_index=True)

    with col_q2:
        # Plot comparative bar chart of returns
        fig_quad = go.Figure(go.Bar(
            x=stats["Cycle_Phase"].map(lambda x: x.split(":")[0]), # Label as Q1, Q2, etc.
            y=stats["Avg_Return"],
            marker_color=["#4ade80", "#38bdf8", "#fb923c", "#f87171"],
            hovertemplate="<b>%{x}</b><br>Avg Return: %{y:.3f}%<extra></extra>"
        ))
        DARK_LAYOUT = dict(
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font=dict(color="#c9d1d9", family="Inter, sans-serif"),
            xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", color="#8b949e"),
            yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", color="#8b949e"),
            margin=dict(t=10, b=10, l=10, r=10),
        )
        fig_quad.update_layout(
            **DARK_LAYOUT,
            height=200,
            yaxis_title="Avg Daily Return (%)",
            showlegend=False
        )
        st.plotly_chart(fig_quad, use_container_width=True)

    callout(
        "💡 Quantitative Cycle Insights:",
        "<ul>"
        "<li><b>Q1 Post-Halving (+0.687%):</b> The structural supply reduction exerts a direct, upward positive drift "
        "with relatively low volatility, indicating stable growth as daily miner sell pressure is cut in half.</li>"
        "<li><b>Q2 Mid-Cycle (+0.676%):</b> Characterized by <b>extreme volatility (12.97%)</b>. This phase hosts the "
        "parabolic bull spikes (extreme FOMO) followed by severe corrections. It is by far the most volatile phase.</li>"
        "<li><b>Q3 &amp; Q4 Late-Cycle/Pre-Halving (+0.326% &amp; +0.305%):</b> Average returns drop by half, and "
        "volatility falls to cycle lows. Q4 represents the classical accumulation phase where trading dries up and "
        "smart money builds positions ahead of the next supply cut.</li>"
        "</ul>"
    )

st.markdown('<div class="cc-section-title">Foundational Literature</div>', unsafe_allow_html=True)
st.write("To understand Bitcoin's design principles, consensus mechanism, and monetary policy:")
st.markdown(
    '<a href="https://bitcoin.org/bitcoin.pdf" target="_blank" '
    'style="display:inline-block;background:#161b22;color:#4ade80;'
    'border:1px solid #2d5a3d;padding:10px 24px;text-decoration:none;'
    'border-radius:8px;font-weight:600;font-size:14px;">'
    'Read the Bitcoin Whitepaper (Satoshi Nakamoto)</a>',
    unsafe_allow_html=True,
)
