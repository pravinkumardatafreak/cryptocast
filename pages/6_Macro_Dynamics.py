import pandas as pd
import streamlit as st

# Set page config
st.set_page_config(
    page_title="CryptoCast | Macro Dynamics",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

def callout(title, body_html, warn=False):
    cls = "cc-callout warn" if warn else "cc-callout"
    st.markdown(
        f'<div class="{cls}"><h4>{title}</h4>{body_html}</div>',
        unsafe_allow_html=True,
    )

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
