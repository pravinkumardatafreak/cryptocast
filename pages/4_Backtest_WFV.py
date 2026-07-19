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
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.streamlit_utils import inject_custom_css, card, callout, DARK_LAYOUT
inject_custom_css()

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(PROJECT_DIR, "model_comparison_results.csv")

# CSS Styles

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
    "2-Fold Halving-Aligned Expanding Window Setup",
    "<ul>"
    "<li><b>Fold 1 (Halving Epoch 2):</b> Train 2010 → 2016-07-09 | Test 2016-07-09 → 2020-05-11 (~4 years: ICO mania & consolidation)</li>"
    "<li><b>Fold 2 (Halving Epoch 3):</b> Train 2010 → 2020-05-11 | Test 2020-05-11 → 2024-03-24 (~4 years: COVID crash, parabolic bull run, bear market & recovery)</li>"
    "</ul>"
    "<p style='margin-top:10px;color:#8b949e;font-size:12px;'>Each test window covers exactly one complete Bitcoin halving epoch — the protocol's natural 4-year market cycle. "
    "This is far more economically meaningful than arbitrary 2-year splits.</p>"
)

if df_wfv is not None:
    NUM_FOLDS = 2  # Halving-aligned: 2 complete epochs
    col_w1, col_w2 = st.columns(2)
    selected_model = col_w1.selectbox("Select Model for WFV Analysis", ["RNN", "1D-CNN", "LSTM", "Transformer", "PatchTST"])
    selected_horizon = col_w2.selectbox("Select Horizon for WFV Analysis", ["1D", "3D", "7D"])

    FOLD_LABELS = [
        "Fold 1 — Epoch 2 (2016–2020)",
        "Fold 2 — Epoch 3 (2020–2024)",
    ]

    # Display table for chosen combination across folds
    fold_data = []
    for f_idx in range(NUM_FOLDS):
        if f_idx >= len(df_wfv[selected_model]):
            break
        metrics = df_wfv[selected_model][f_idx][selected_horizon]
        fold_data.append({
            "Fold": FOLD_LABELS[f_idx],
            "MAE (USD)": f"${metrics['MAE']:,.2f}",
            "RMSE (USD)": f"${metrics['RMSE']:,.2f}",
            "MAPE (%)": f"{metrics['MAPE']:.2f}%",
            "raw_mape": metrics['MAPE']
        })

    # Calculate Average
    valid_n = len(fold_data)
    avg_mae  = np.mean([df_wfv[selected_model][f][selected_horizon]['MAE']  for f in range(valid_n)])
    avg_rmse = np.mean([df_wfv[selected_model][f][selected_horizon]['RMSE'] for f in range(valid_n)])
    avg_mape = np.mean([df_wfv[selected_model][f][selected_horizon]['MAPE'] for f in range(valid_n)])
    
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
    for mdl in ["RNN", "1D-CNN", "LSTM", "Transformer", "PatchTST"]:
        for hz in ["1D", "3D", "7D"]:
            if mdl in df_wfv:
                num_folds = len(df_wfv[mdl])
                mapes = [df_wfv[mdl][f_idx][hz]['MAPE'] for f_idx in range(num_folds)]
                if mapes:
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
