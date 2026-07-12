import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Set page config
st.set_page_config(
    page_title="CryptoCast | Backtest (WFV)",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(PROJECT_DIR, "model_comparison_results.csv")

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

def callout(title, body_html):
    st.markdown(
        f'<div class="cc-callout"><h4>{title}</h4>{body_html}</div>',
        unsafe_allow_html=True,
    )

@st.cache_data(ttl=0)
def load_wfv_results():
    wfv_path = os.path.join(PROJECT_DIR, "wfv_results.json")
    if os.path.exists(wfv_path):
        with open(wfv_path, "r") as f:
            return json.load(f)
    return None

df_wfv = load_wfv_results()

st.markdown('<div class="cc-eyebrow">Backtesting</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-title">Walk-Forward Validation (WFV)</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-subtitle">Audit model generalisation and error rates across expanding historical market phases</div>', unsafe_allow_html=True)

st.markdown('<div class="cc-section-title">Walk-Forward Validation Theory</div>', unsafe_allow_html=True)
st.write(
    "Time-series models are highly sensitive to market regimes. Evaluating a model "
    "on a single test period can be misleading. Walk-Forward Validation (expanding window) "
    "evaluates model performance across multiple historical phases."
)

callout(
    "3-Fold Expanding Window Setup",
    "<ul>"
    "<li><b>Fold 1:</b> Train on 2010-2018 -> Test on 2019-2020 (Consolidation phase)</li>"
    "<li><b>Fold 2:</b> Train on 2010-2020 -> Test on 2020-2022 (Bull run & crash)</li>"
    "<li><b>Fold 3:</b> Train on 2010-2022 -> Test on 2022-2024 (Bear market & recovery)</li>"
    "</ul>"
)

if df_wfv is not None:
    col_w1, col_w2 = st.columns(2)
    selected_model = col_w1.selectbox("Select Model for WFV Analysis", ["RNN", "1D-CNN", "LSTM", "Transformer"])
    selected_horizon = col_w2.selectbox("Select Horizon for WFV Analysis", ["1D", "3D", "7D"])
    
    # Display table for chosen combination across folds
    fold_data = []
    for f_idx in range(3):
        metrics = df_wfv[selected_model][f_idx][selected_horizon]
        fold_data.append({
            "Fold": f"Fold {f_idx + 1}",
            "MAE (USD)": f"${metrics['MAE']:,.2f}",
            "RMSE (USD)": f"${metrics['RMSE']:,.2f}",
            "MAPE (%)": f"{metrics['MAPE']:.2f}%",
            "raw_mape": metrics['MAPE']
        })
        
    # Calculate Average
    avg_mae = np.mean([df_wfv[selected_model][f_idx][selected_horizon]['MAE'] for f_idx in range(3)])
    avg_rmse = np.mean([df_wfv[selected_model][f_idx][selected_horizon]['RMSE'] for f_idx in range(3)])
    avg_mape = np.mean([df_wfv[selected_model][f_idx][selected_horizon]['MAPE'] for f_idx in range(3)])
    
    fold_data.append({
        "Fold": "Average",
        "MAE (USD)": f"${avg_mae:,.2f}",
        "RMSE (USD)": f"${avg_rmse:,.2f}",
        "MAPE (%)": f"{avg_mape:.2f}%",
        "raw_mape": avg_mape
    })
    
    df_fold_table = pd.DataFrame(fold_data)
    st.markdown(f"**Performance of {selected_model} on {selected_horizon} Horizon Across Folds**")
    st.dataframe(df_fold_table.drop(columns=["raw_mape"]), use_container_width=True, hide_index=True)
    
    # Plot Plotly bar chart comparing folds
    fig_wfv = go.Figure()
    fig_wfv.add_trace(go.Bar(
        x=[d["Fold"] for d in fold_data[:-1]],
        y=[d["raw_mape"] for d in fold_data[:-1]],
        marker_color=["#4ade80", "#38bdf8", "#fb923c"],
        hovertemplate="<b>%{x}</b><br>MAPE: %{y:.2f}%<extra></extra>"
    ))
    fig_wfv.add_hline(y=avg_mape, line_dash="dash", line_color="#f87171",
                      annotation_text=f"Average: {avg_mape:.2f}%", annotation_font_color="#f87171")
    fig_wfv.update_layout(
        **DARK_LAYOUT,
        title=f"Walk-Forward MAPE (%) Comparison for {selected_model} ({selected_horizon})",
        yaxis_title="MAPE (%)",
        height=350,
        showlegend=False
    )
    st.plotly_chart(fig_wfv, use_container_width=True)
    
    # Compare all models' average MAPEs across all folds
    st.markdown('<div class="cc-section-title">Model Stability Comparison (Average MAPE across Folds)</div>', unsafe_allow_html=True)
    comparison_rows = []
    for mdl in ["RNN", "1D-CNN", "LSTM", "Transformer"]:
        for hz in ["1D", "3D", "7D"]:
            mapes = [df_wfv[mdl][f_idx][hz]['MAPE'] for f_idx in range(3)]
            comparison_rows.append({
                "Model": mdl,
                "Horizon": hz,
                "Avg WFV MAPE (%)": np.mean(mapes)
            })
    df_comp = pd.DataFrame(comparison_rows)
    fig_comp = px.bar(
        df_comp, x="Model", y="Avg WFV MAPE (%)", color="Horizon", barmode="group",
        color_discrete_map={"1D": "#4ade80", "3D": "#38bdf8", "7D": "#fb923c"}
    )
    fig_comp.update_layout(
        **DARK_LAYOUT,
        title="Stability Comparison: Average WFV MAPE (%) by Model and Horizon",
        yaxis_title="Average WFV MAPE (%)",
        height=400,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9"))
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    
else:
    st.info("Walk-Forward Validation metrics not found. Run step4_wfv.py to generate them.")
