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
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.streamlit_utils import inject_custom_css, card, callout, DARK_LAYOUT
inject_custom_css()

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
VIZ_DIR     = os.path.join(PROJECT_DIR, "visualizations")

# CSS Styles

st.markdown('<div class="cc-eyebrow">Auditing</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-title">Model Diagnostics</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-subtitle">Inspect loss curves and check exact actual vs. predicted deviations for individual models</div>', unsafe_allow_html=True)

d1, d2 = st.columns(2)
horz = d1.selectbox("Horizon", ["1D", "3D", "7D"])
mdl  = d2.selectbox("Model",   ["1D-CNN", "RNN", "LSTM", "Transformer", "PatchTST"])

json_path = os.path.join(RESULTS_DIR, f"{mdl}_{horz}.json")
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        metrics_data = json.load(f)

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (USD)",  f"${metrics_data['MAE']:,.0f}")
    m2.metric("RMSE (USD)", f"${metrics_data['RMSE']:,.0f}")
    m3.metric("MAPE",       f"{metrics_data['MAPE']:.2f}%")

    import pandas as pd
    import numpy as np

    # ── Reconstruct real dates for the test set ─────────────────────────────
    DATA_PATH = os.path.join(PROJECT_DIR, "data", "btc_data.csv")
    SEQ_LENGTH = 60

    @st.cache_data(ttl=0)
    def get_test_dates(horizon_str):
        df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
        split = int(len(df) * 0.80)
        test_df = df.iloc[split:]
        horizon_map = {"1D": 1, "3D": 3, "7D": 7}
        h = horizon_map.get(horizon_str, 1)
        # Each sample i predicts SEQ_LENGTH+h-1 days into test_df
        sample_dates = [
            test_df.index[i + SEQ_LENGTH + h - 1]
            for i in range(len(metrics_data["y_test"]))
            if (i + SEQ_LENGTH + h - 1) < len(test_df)
        ]
        return sample_dates

    test_dates = get_test_dates(horz)
    n_pts = min(len(test_dates), len(metrics_data["y_test"]), len(metrics_data["y_pred"]))
    dates_ser  = pd.DatetimeIndex(test_dates[:n_pts])
    y_actual   = metrics_data["y_test"][:n_pts]
    y_pred     = metrics_data["y_pred"][:n_pts]

    # ── Embedded US CPI YoY% (monthly, 2020-2024) ──────────────────────────
    # Source: U.S. Bureau of Labor Statistics (CPI-U, All Items, YoY %)
    # Used as a proxy for global inflation pressure
    CPI_MONTHLY = {
        "2020-01": 2.5, "2020-02": 2.3, "2020-03": 1.5, "2020-04": 0.3,
        "2020-05": 0.1, "2020-06": 0.6, "2020-07": 1.0, "2020-08": 1.3,
        "2020-09": 1.4, "2020-10": 1.2, "2020-11": 1.2, "2020-12": 1.4,
        "2021-01": 1.4, "2021-02": 1.7, "2021-03": 2.6, "2021-04": 4.2,
        "2021-05": 5.0, "2021-06": 5.4, "2021-07": 5.4, "2021-08": 5.3,
        "2021-09": 5.4, "2021-10": 6.2, "2021-11": 6.8, "2021-12": 7.0,
        "2022-01": 7.5, "2022-02": 7.9, "2022-03": 8.5, "2022-04": 8.3,
        "2022-05": 8.6, "2022-06": 9.1, "2022-07": 8.5, "2022-08": 8.3,
        "2022-09": 8.2, "2022-10": 7.7, "2022-11": 7.1, "2022-12": 6.5,
        "2023-01": 6.4, "2023-02": 6.0, "2023-03": 5.0, "2023-04": 4.9,
        "2023-05": 4.0, "2023-06": 3.0, "2023-07": 3.2, "2023-08": 3.7,
        "2023-09": 3.7, "2023-10": 3.2, "2023-11": 3.1, "2023-12": 3.4,
        "2024-01": 3.1, "2024-02": 3.2, "2024-03": 3.5,
    }

    def get_inflation(dt):
        key = dt.strftime("%Y-%m")
        return CPI_MONTHLY.get(key, None)

    inflation_vals = [get_inflation(d) for d in dates_ser]

    # ── Build chart with dual y-axis ────────────────────────────────────────
    st.markdown('<div class="cc-section-title">Actual vs Predicted — with Global Inflation Context</div>',
                unsafe_allow_html=True)

    custom_hover_actual = [
        (f"<b>{d.strftime('%b %d, %Y')}</b><br>"
         f"Actual: <b>${y:,.0f}</b><br>"
         f"CPI Inflation: <b>{inf:.1f}%</b>" if inf is not None
         else f"<b>{d.strftime('%b %d, %Y')}</b><br>Actual: <b>${y:,.0f}</b><br>CPI: N/A")
        for d, y, inf in zip(dates_ser, y_actual, inflation_vals)
    ]
    custom_hover_pred = [
        (f"<b>{d.strftime('%b %d, %Y')}</b><br>"
         f"Predicted: <b>${y:,.0f}</b><br>"
         f"CPI Inflation: <b>{inf:.1f}%</b>" if inf is not None
         else f"<b>{d.strftime('%b %d, %Y')}</b><br>Predicted: <b>${y:,.0f}</b><br>CPI: N/A")
        for d, y, inf in zip(dates_ser, y_pred, inflation_vals)
    ]

    fig_avp = go.Figure()

    # Actual price
    fig_avp.add_trace(go.Scatter(
        x=dates_ser, y=y_actual, name="Actual BTC Price",
        line=dict(color="#4ade80", width=1.8),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=custom_hover_actual,
        yaxis="y1",
    ))
    # Predicted price
    fig_avp.add_trace(go.Scatter(
        x=dates_ser, y=y_pred, name="Predicted BTC Price",
        line=dict(color="#fb923c", width=1.5, dash="dash"),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=custom_hover_pred,
        yaxis="y1",
    ))

    # Inflation area on secondary axis
    infl_dates = [d for d, v in zip(dates_ser, inflation_vals) if v is not None]
    infl_vals  = [v for v in inflation_vals if v is not None]

    fig_avp.add_trace(go.Scatter(
        x=infl_dates, y=infl_vals, name="US CPI YoY% (Global Inflation Proxy)",
        line=dict(color="#38bdf8", width=1.2, dash="dot"),
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.06)",
        hovertemplate="<b>%{x|%b %Y}</b><br>CPI Inflation: <b>%{y:.1f}%</b><extra></extra>",
        yaxis="y2",
    ))

    # Add annotation for peak inflation
    fig_avp.add_annotation(
        x="2022-06-01", y=9.1, yref="y2",
        text="Peak: 9.1%<br>Jun 2022",
        showarrow=True, arrowhead=2,
        arrowcolor="#38bdf8", font=dict(color="#38bdf8", size=11),
        bgcolor="rgba(13,17,23,0.8)", bordercolor="#38bdf8",
    )

    fig_avp.update_layout(**DARK_LAYOUT)
    fig_avp.update_layout(
        title=dict(
            text=f"{mdl} ({horz} horizon): Actual vs Predicted | with Monthly CPI Inflation",
            font=dict(color="#e6edf3", size=14),
        ),
        xaxis=dict(title="Date", gridcolor="#21262d", color="#8b949e"),
        yaxis=dict(
            title="BTC Price (USD)", gridcolor="#21262d",
            color="#8b949e", side="left",
        ),
        yaxis2=dict(
            title=dict(text="CPI Inflation YoY (%)", font=dict(color="#38bdf8")),
            overlaying="y", side="right",
            color="#38bdf8", showgrid=False, tickformat=".1f",
            tickfont=dict(color="#38bdf8"),
        ),
        hovermode="x unified",
        height=460,
        legend=dict(
            bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363d",
            borderwidth=1, font=dict(color="#c9d1d9"), x=0.01, y=0.99,
        ),
    )
    st.plotly_chart(fig_avp, use_container_width=True)

    st.markdown(
        '<p style="color:#8b949e;font-size:12px;margin-top:-12px;">'
        'CPI data: U.S. Bureau of Labor Statistics, CPI-U All Items YoY% — used as global inflation proxy. '
        'Source: BLS.gov. High inflation periods (2021-Q4 to 2022-Q3) correlate with extreme BTC volatility.</p>',
        unsafe_allow_html=True,
    )

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
    
    # ── Residual Diagnostics ───────────────────────────────────────────────
    st.markdown('<div class="cc-section-title">Residual Diagnostics (Statistical Validation)</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e;font-size:14px;margin-bottom:15px;">'
                'Analyzing the residuals (Actual - Predicted). If a model has extracted all available signal, '
                'the residuals should look like random "White Noise" with no distinct autocorrelation patterns.</p>', 
                unsafe_allow_html=True)
                
    residuals = np.array(y_actual) - np.array(y_pred)
    
    r1, r2 = st.columns(2)
    
    with r1:
        # Residual Distribution
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=residuals, nbinsx=50, name="Residuals", 
            marker_color="#8b5cf6", opacity=0.75
        ))
        fig_dist.update_layout(
            **DARK_LAYOUT,
            title=dict(text="Residual Distribution", font=dict(color="#e6edf3")),
            xaxis_title="Error (USD)", yaxis_title="Count",
            height=300, showlegend=False
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with r2:
        # Autocorrelation (ACF)
        try:
            from statsmodels.tsa.stattools import acf
            # Calculate ACF for up to 40 lags
            acf_vals, confint = acf(residuals, nlags=40, alpha=0.05)
            lags = np.arange(len(acf_vals))
            
            fig_acf = go.Figure()
            # Confidence intervals
            fig_acf.add_trace(go.Scatter(x=lags, y=confint[:,1]-acf_vals, mode='lines', line_color='rgba(255,255,255,0)', showlegend=False))
            fig_acf.add_trace(go.Scatter(x=lags, y=confint[:,0]-acf_vals, mode='lines', line_color='rgba(255,255,255,0)', fill='tonexty', fillcolor='rgba(56,189,248,0.2)', showlegend=False))
            
            # Stem plot for ACF
            for i in range(len(lags)):
                fig_acf.add_trace(go.Scatter(x=[lags[i], lags[i]], y=[0, acf_vals[i]], mode='lines', line=dict(color='#38bdf8', width=2), showlegend=False))
            fig_acf.add_trace(go.Scatter(x=lags, y=acf_vals, mode='markers', marker=dict(color='#38bdf8', size=6), showlegend=False))
            
            fig_acf.update_layout(
                **DARK_LAYOUT,
                title=dict(text="Autocorrelation (ACF) of Residuals", font=dict(color="#e6edf3")),
                xaxis_title="Lag (Days)", yaxis_title="ACF",
                height=300
            )
            st.plotly_chart(fig_acf, use_container_width=True)
        except ImportError:
            st.warning("Please install statsmodels (`pip install statsmodels`) to view the ACF plot.")
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
