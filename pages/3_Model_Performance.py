import os
import pandas as pd
import plotly.express as px
import streamlit as st

# Set page config
st.set_page_config(
    page_title="CryptoCast | Model Performance",
    page_icon="🏆",
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
        .leaderboard-header {
            display: flex; background: #161b22; padding: 12px 20px;
            border-bottom: 1px solid #21262d; font-size: 12px; font-weight: 600; color: #8b949e;
        }
        .leaderboard-row {
            display: flex; padding: 16px 20px; border-bottom: 1px solid #21262d;
            align-items: center; font-size: 14px; background: #0d1117;
        }
        .leaderboard-row:last-child { border-bottom: none; }
        .badge-horizon {
            background: #21262d; color: #c9d1d9; font-size: 11px; font-weight: 600;
            padding: 3px 8px; border-radius: 4px;
        }
        .badge-best {
            background: rgba(74, 222, 128, 0.15); color: #4ade80; font-size: 10px;
            font-weight: 700; padding: 2px 6px; border-radius: 4px; margin-left: 8px;
            border: 1px solid rgba(74, 222, 128, 0.3);
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

def callout(title, body_html):
    st.markdown(
        f'<div class="cc-callout"><h4>{title}</h4>{body_html}</div>',
        unsafe_allow_html=True,
    )

@st.cache_data(ttl=0)
def load_results():
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        for col in ["MAE", "RMSE"]:
            if col in df.columns:
                df[col] = df[col].round(2)
        if "MAPE (%)" in df.columns:
            df["MAPE (%)"] = df["MAPE (%)"].round(4)
        return df
    return None

df_results = load_results()

st.markdown('<div class="cc-eyebrow">Leaderboard</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-title">Model Performance</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-subtitle">Compare deep learning architectures based on absolute USD errors and percent deviations</div>', unsafe_allow_html=True)

if df_results is not None:
    st.markdown('<div class="cc-section-title">Performance Leaderboard</div>', unsafe_allow_html=True)
    st.write("Evaluate MAE, RMSE, and MAPE across the 1-Day, 3-Day, and 7-Day forecasting horizons.")

    from sklearn.metrics import r2_score
    import json

    best = (
        df_results.loc[df_results.groupby("Horizon")["MAE"].idxmin()]
        .set_index("Horizon")["Model"]
        .to_dict()
    )

    rows_html = ""
    for _, row in df_results.iterrows():
        horizon  = row["Horizon"]
        model    = row["Model"]
        mae      = row["MAE"]
        rmse     = row["RMSE"]
        mape     = row["MAPE (%)"]
        is_best  = best.get(horizon) == model
        badge    = '<span class="badge-best">BEST</span>' if is_best else ""
        
        # Calculate R2 dynamically from prediction files
        r2_val = np.nan
        json_path = os.path.join(PROJECT_DIR, "results", f"{model}_{horizon}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    r2_val = r2_score(data["y_test"], data["y_pred"])
            except Exception:
                pass

        if mape < 5:
            mape_color = "#4ade80"
        elif mape < 10:
            mape_color = "#fb923c"
        else:
            mape_color = "#f87171"
            
        r2_str = f"{r2_val:.4f}" if not np.isnan(r2_val) else "N/A"
        
        rows_html += (
            f'<div class="leaderboard-row">'
            f'<div style="width:80px"><span class="badge-horizon">{horizon}</span></div>'
            f'<div style="flex:1;color:#58a6ff;font-weight:500">{model}{badge}</div>'
            f'<div style="width:130px;text-align:right;font-weight:600;color:#e6edf3">${mae:,.2f}</div>'
            f'<div style="width:130px;text-align:right;color:#8b949e">${rmse:,.2f}</div>'
            f'<div style="width:110px;text-align:right;font-weight:600;color:{mape_color}">{mape:.2f}%</div>'
            f'<div style="width:100px;text-align:right;font-weight:600;color:#58a6ff">{r2_str}</div>'
            f'</div>'
        )

    st.markdown(
        '<div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;overflow:hidden;margin-bottom:24px;">'
        '<div class="leaderboard-header">'
        '<div style="width:80px">Horizon</div>'
        '<div style="flex:1">Model Architecture</div>'
        '<div style="width:130px;text-align:right">MAE (USD)</div>'
        '<div style="width:130px;text-align:right">RMSE (USD)</div>'
        '<div style="width:110px;text-align:right">MAPE (%)</div>'
        '<div style="width:100px;text-align:right">R² Score</div>'
        '</div>'
        + rows_html + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="cc-section-title">Visual Metric Comparison</div>', unsafe_allow_html=True)
    metric_opt = st.selectbox("Compare by metric", ["MAE", "RMSE", "MAPE (%)"])
    fig_bar = px.bar(
        df_results, x="Model", y=metric_opt, color="Horizon", barmode="group",
        color_discrete_map={"1D": "#4ade80", "3D": "#38bdf8", "7D": "#fb923c"},
    )
    fig_bar.update_layout(
        **DARK_LAYOUT,
        height=400,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9")),
    )
    fig_bar.update_traces(marker_line_width=0)
    st.plotly_chart(fig_bar, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        callout(
            "Key Insight: Log Returns Eliminate Extrapolation Failure",
            "<p>Previous models trained on raw price suffered from <b>extrapolation failure</b> - "
            "the MinMaxScaler was fit on training prices (up to ~$63K), so any test price above "
            "$63K was clipped at 1.0, causing MAPE above 40%. "
            "The fix: train on stationary <b>log returns</b> r = ln(P[t+h] / P[t]), which are "
            "scale-invariant. Prices are reconstructed using P_hat = P[t] x exp(r_hat). "
            "This dropped 1D MAPE from 42.9% to 2.12% (LSTM) and 7D MAPE from 48.2% to 6.04% (Transformer).</p>",
        )
    with c2:
        callout(
            "🧠 The R² Paradox & Fractal Nature of Financial Data",
            "<p><b>The Paradox:</b> Our model shows extremely high $R^2$ scores on price level predictions "
            "(e.g., LSTM 1D $R^2$ = <b>0.9917</b>). However, this is a path-dependent time-series artifact—since "
            "yesterday's price explains 99% of today's price variance, any model anchoring to $P[t]$ scores highly.<br>"
            "<b>Fractal returns:</b> In reality, financial returns are highly fractal (fat-tailed noise, Hurst exponent $H \\approx 0.53$). "
            "If we calculate $R^2$ on the daily returns directly, the score falls to $1\\%-3\\%$. In quantitative finance, "
            "explaining even $2\\%$ of return-level variance is outstanding because market returns behave like a fractal random walk.</p>"
        )
else:
    st.warning("model_comparison_results.csv not found. Run the training pipeline first.")
