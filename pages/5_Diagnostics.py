import os
import json
import plotly.graph_objects as go
import streamlit as st

# Set page config
st.set_page_config(
    page_title="CryptoCast | Diagnostics",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
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

st.markdown('<div class="cc-eyebrow">Auditing</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-title">Model Diagnostics</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-subtitle">Inspect loss curves and check exact actual vs. predicted deviations for individual models</div>', unsafe_allow_html=True)

d1, d2 = st.columns(2)
horz = d1.selectbox("Horizon", ["1D", "3D", "7D"])
mdl  = d2.selectbox("Model",   ["1D-CNN", "RNN", "LSTM", "Transformer"])

json_path = os.path.join(RESULTS_DIR, f"{mdl}_{horz}.json")
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        metrics_data = json.load(f)

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (USD)",  f"${metrics_data['MAE']:,.0f}")
    m2.metric("RMSE (USD)", f"${metrics_data['RMSE']:,.0f}")
    m3.metric("MAPE",       f"{metrics_data['MAPE']:.2f}%")

    st.markdown('<div class="cc-section-title">Actual vs Predicted</div>', unsafe_allow_html=True)
    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(
        y=metrics_data["y_test"], name="Actual",
        line=dict(color="#4ade80", width=1.5)
    ))
    fig_avp.add_trace(go.Scatter(
        y=metrics_data["y_pred"], name="Predicted",
        line=dict(color="#fb923c", width=1.5, dash="dash")
    ))
    fig_avp.update_layout(
        **DARK_LAYOUT,
        title=dict(text=f"{mdl} - {horz} horizon: Actual vs Predicted",
                   font=dict(color="#e6edf3")),
        xaxis_title="Test set time steps",
        yaxis_title="Price (USD)",
        hovermode="x unified", height=380,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9")),
    )
    st.plotly_chart(fig_avp, use_container_width=True)

    st.markdown('<div class="cc-section-title">Training Loss Curve</div>', unsafe_allow_html=True)
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        y=metrics_data["history"]["loss"], name="Train Loss",
        line=dict(color="#38bdf8", width=1.5)
    ))
    fig_loss.add_trace(go.Scatter(
        y=metrics_data["history"]["val_loss"], name="Val Loss",
        line=dict(color="#fb923c", width=1.5, dash="dash")
    ))
    fig_loss.update_layout(
        **DARK_LAYOUT,
        title=dict(text=f"{mdl} - {horz} horizon: Loss Curve",
                   font=dict(color="#e6edf3")),
        xaxis_title="Epoch", yaxis_title="MSE Loss",
        height=320,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9")),
    )
    st.plotly_chart(fig_loss, use_container_width=True)
else:
    st.info(f"No saved results for {mdl} / {horz}. Run the training pipeline first.")
    i1, i2 = st.columns(2)
    with i1:
        p = os.path.join(VIZ_DIR, f"09_loss_curves_{horz}.png")
        if os.path.exists(p):
            st.image(p, caption=f"Loss curves - {horz}", use_container_width=True)
    with i2:
        p = os.path.join(VIZ_DIR, f"10_actual_vs_predicted_{horz}.png")
        if os.path.exists(p):
            st.image(p, caption=f"Actual vs Predicted - {horz}", use_container_width=True)
