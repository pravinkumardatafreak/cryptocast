import streamlit as st

def inject_custom_css():
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
            .cc-callout.warn { border-left-color: #fb923c; }
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

def callout(title, body_html, warn=False):
    cls = "cc-callout warn" if warn else "cc-callout"
    st.markdown(
        f'<div class="{cls}"><h4>{title}</h4>{body_html}</div>',
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
