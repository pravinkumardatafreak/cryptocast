"""
CryptoCast - Bitcoin Price Forecasting Dashboard
=================================================
Interactive presentation dashboard for the CryptoCast capstone project.

Usage:
    streamlit run app.py
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="CryptoCast | Bitcoin Price Forecasting",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(PROJECT_DIR, "data", "btc_data.csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
RESULTS_CSV = os.path.join(PROJECT_DIR, "model_comparison_results.csv")
VIZ_DIR     = os.path.join(PROJECT_DIR, "visualizations")

# -- Dark Gemini-style theme ---------------------------------------------------
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

        [data-testid="stTabs"] button {
            color: #8b949e !important;
            font-weight: 500;
            font-size: 14px;
            border-bottom: 2px solid transparent;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #4ade80 !important;
            border-bottom: 2px solid #4ade80 !important;
        }
        [data-testid="stTabs"] { border-bottom: 1px solid #21262d; }

        [data-testid="stMetric"] {
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 10px;
            padding: 14px 18px;
        }
        [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 12px; }
        [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 26px; font-weight: 700; }

        [data-testid="stSelectbox"] > div > div {
            background: #161b22 !important;
            border: 1px solid #30363d !important;
            color: #e6edf3 !important;
        }

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
        .cc-card h4 {
            font-size: 11px; font-weight: 600; color: #8b949e;
            text-transform: uppercase; letter-spacing: 0.06em; margin: 0 0 8px 0;
        }
        .cc-card .cc-value { font-size: 24px; font-weight: 700; color: #e6edf3; margin: 0; }
        .cc-card .cc-detail { font-size: 12px; color: #6e7681; margin-top: 4px; }

        .cc-callout {
            border: 1px solid #21262d; border-left: 3px solid #4ade80;
            background: #161b22; border-radius: 8px;
            padding: 16px 20px; margin: 14px 0;
        }
        .cc-callout.warn { border-left-color: #f97316; background: #1a1610; }
        .cc-callout h4 { margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #e6edf3; }
        .cc-callout p, .cc-callout li { font-size: 14px; color: #c9d1d9; line-height: 1.65; margin: 0; }
        .cc-callout ul { margin: 6px 0 0 0; padding-left: 20px; }

        .cc-tag {
            display: inline-block; background: #21262d; color: #4ade80;
            border: 1px solid #2d5a3d; font-size: 12px; font-weight: 500;
            padding: 4px 12px; border-radius: 20px; margin-right: 8px; margin-bottom: 8px;
        }
        .leaderboard-row {
            display: flex; align-items: center; padding: 12px 16px;
            border-bottom: 1px solid #21262d; font-size: 14px; color: #c9d1d9;
        }
        .leaderboard-row:hover { background: #1c2128; }
        .leaderboard-header {
            display: flex; align-items: center; padding: 10px 16px;
            background: #161b22; border-bottom: 1px solid #30363d;
            font-size: 11px; font-weight: 600; color: #8b949e;
            text-transform: uppercase; letter-spacing: 0.05em;
            border-radius: 8px 8px 0 0;
        }
        .badge-best {
            display: inline-block; background: #1a3d2b; color: #4ade80;
            font-size: 10px; font-weight: 700; padding: 2px 8px;
            border-radius: 4px; margin-left: 8px; border: 1px solid #2d5a3d;
        }
        .badge-horizon {
            display: inline-block; width: 32px; text-align: center;
            background: #21262d; color: #58a6ff; font-size: 11px;
            font-weight: 700; padding: 3px 6px; border-radius: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -- Plotly dark layout defaults -----------------------------------------------
DARK_LAYOUT = dict(
    plot_bgcolor="#0d1117",
    paper_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", color="#8b949e"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", color="#8b949e"),
    margin=dict(t=30, b=30, l=10, r=10),
)


# -- Utility components --------------------------------------------------------
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


# -- Data loaders (ttl=0 forces fresh read - no stale cache) -------------------
@st.cache_data(ttl=0)
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    return None


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


df_raw     = load_data()
df_results = load_results()

# -- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### CryptoCast")
    st.caption("Multi-Horizon Bitcoin Price Forecasting")
    st.divider()
    st.markdown(
        "**Models:** 1D-CNN / RNN / LSTM / Transformer\n\n"
        "**Horizons:** 1D / 3D / 7D\n\n"
        "**Data:** ~5,000 daily BTC records (2010-2024)"
    )
    st.divider()
   
    

# -- Header --------------------------------------------------------------------
st.markdown('<div class="cc-eyebrow">Capstone Project</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-title">CryptoCast: Bitcoin Price Forecasting</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cc-subtitle">Comparing deep learning architectures for '
    '1-day, 3-day, and 7-day BTC price forecasts</div>',
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "Overview",
    "Exploratory Analysis",
    "Seasonality Analysis",
    "Model Performance",
    "Diagnostics",
    "Macro & Halving Dynamics",
])

# =============================================================================
# Tab 1: Overview
# =============================================================================
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Dataset", "4,993 records", "Daily BTC prices, 2010-2024")
    with c2:
        card("Sequence Length", "60 days", "Sliding window input")
    with c3:
        card("Forecast Horizons", "1D / 3D / 7D", "Multi-step prediction")
    with c4:
        card("Validation", "Time-based split", "80 / 20, ")

    st.markdown('<div class="cc-section-title">Problem Statement</div>', unsafe_allow_html=True)
    st.write(
        "Bitcoin prices are highly volatile and shaped by complex, non-linear temporal dynamics. "
        "This project builds and compares deep learning architectures that learn from historical "
        "price sequences to forecast BTC prices across three horizons - supporting use cases such as "
        "short-term trading signals, multi-horizon algorithmic decision-making, and volatility-aware "
        "risk management."
    )

    st.markdown('<div class="cc-section-title">Model Architectures</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        callout("1D-CNN",
            "<p>Convolutional filters extract local short-term patterns. "
            "Fast to train, strong on short horizons.</p>")
        callout("RNN",
            "<p>Baseline sequential model. Efficient, but limited by vanishing gradients over "
            "long sequences. Best performer on the 1D and 3D horizons.</p>")
    with a2:
        callout("LSTM",
            "<p>Gated memory cells retain long-range dependencies, reducing the vanishing gradient "
            "problem seen in vanilla RNNs.</p>")
        callout("Transformer",
            "<p>Self-attention captures global context across the full 60-day sequence. "
            "Best performer on the 7D horizon.</p>")

    st.markdown('<div class="cc-section-title">Business Relevance</div>', unsafe_allow_html=True)
    st.markdown(
        '<span class="cc-tag">Trading signal generation</span>'
        '<span class="cc-tag">Algorithmic multi-horizon decisions</span>'
        '<span class="cc-tag">Volatility &amp; risk management</span>'
        '<span class="cc-tag">Investment analytics dashboards</span>',
        unsafe_allow_html=True,
    )

# =============================================================================
# Tab 2: Exploratory Analysis
# =============================================================================
with tabs[1]:
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

# =============================================================================
# Tab 3: Seasonality Analysis
# =============================================================================
with tabs[2]:
    if df_raw is not None:
        st.markdown('<div class="cc-section-title">Bitcoin Monthly Returns Heatmap (%)</div>', unsafe_allow_html=True)

        # Show dataset scope clearly
        date_min = df_raw.index.min().strftime("%b %Y")
        date_max = df_raw.index.max().strftime("%b %Y")
        n_years  = df_raw.index.year.nunique()
        st.markdown(
            f'<p style="color:#8b949e;font-size:13px;margin-bottom:16px;">'
            f'All calculations below are derived <b style="color:#4ade80;">exclusively</b> from '
            f'your dataset: <b style="color:#e6edf3;">{date_min} to {date_max}</b> '
            f'({n_years} calendar years, {len(df_raw):,} daily records). '
            f'No external data is used.</p>',
            unsafe_allow_html=True,
        )
        st.write(
            "Bitcoin's performance shows strong monthly seasonality. "
            "Green cells = positive month, Red cells = negative month."
        )

        # Compute monthly returns
        monthly_prices = df_raw["Price"].resample("ME").last()
        monthly_pct    = monthly_prices.pct_change() * 100
        monthly_df     = monthly_pct.to_frame(name="Return")
        monthly_df["Year"]  = monthly_df.index.year
        monthly_df["Month"] = monthly_df.index.month

        month_map  = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                      7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        month_cols = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly_df["Month Name"] = monthly_df["Month"].map(month_map)

        pivot_df = (
            monthly_df.pivot(index="Year", columns="Month", values="Return")
            .rename(columns=month_map)
            .reindex(columns=month_cols)
            .iloc[::-1]
        )

        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index.astype(str),
            colorscale=[
                [0.0,  "rgb(185,28,28)"],
                [0.45, "rgb(100,20,20)"],
                [0.5,  "rgb(30,30,30)"],
                [0.55, "rgb(20,70,35)"],
                [1.0,  "rgb(21,128,61)"],
            ],
            zmid=0,
            text=np.round(pivot_df.values, 1),
            texttemplate="%{text}%",
            hoverongaps=False,
            colorbar=dict(tickfont=dict(color="#c9d1d9"), outlinewidth=0),
        ))
        fig_heat.update_layout(
            **DARK_LAYOUT,
            xaxis_title="Month",
            yaxis_title="Year",
            height=520,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown('<div class="cc-section-title">Monthly Performance Statistics</div>', unsafe_allow_html=True)
        avg_ret  = monthly_df.groupby("Month Name")["Return"].mean().reindex(month_cols)
        win_rate = (
            monthly_df.groupby("Month Name")["Return"]
            .apply(lambda x: (x > 0).sum() / x.notna().sum() * 100)
            .reindex(month_cols)
        )

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("**Average Return (%) by Month**")
            # Green for positive months, red for negative months
            bar_colors = ["#4ade80" if v >= 0 else "#f87171" for v in avg_ret.values]
            fig_avg = go.Figure(go.Bar(
                x=avg_ret.index,
                y=avg_ret.values,
                marker_color=bar_colors,
                marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>Avg Return: %{y:.1f}%<extra></extra>",
            ))
            fig_avg.update_layout(
                **DARK_LAYOUT,
                height=320,
                yaxis_title="Avg Return (%)",
                showlegend=False,
            )
            st.plotly_chart(fig_avg, use_container_width=True)

        with col_m2:
            st.markdown("**Historical Win Rate (%) by Month**")
            # Single clean color for all bars
            fig_win = go.Figure(go.Bar(
                x=win_rate.index,
                y=win_rate.values,
                marker_color="#38bdf8",
                marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>Win Rate: %{y:.1f}%<extra></extra>",
            ))
            fig_win.add_hline(
                y=50, line_dash="dash", line_color="#6e7681",
                annotation_text="50% baseline",
                annotation_font_color="#8b949e",
            )
            fig_win.update_layout(
                **DARK_LAYOUT,
                height=320,
                yaxis_title="Win Rate (%)",
                showlegend=False,
            )
            st.plotly_chart(fig_win, use_container_width=True)

        callout(
            "Key Seasonality Observations (Based on dataset: Aug 2010 - Mar 2024)",
            "<ul>"
            "<li><b>Strongest Months:</b> April (+38.3% avg) and November (+38.7% avg) show the "
            "highest average returns in the dataset. October (+21.0%) also posts consistently strong "
            "results with a 71.4% win rate.</li>"
            "<li><b>Weakest Months:</b> August (-0.1%) and September (-4.8%) are the only two months "
            "with negative average returns. September has the lowest win rate at 35.7% - "
            "meaning it closed positive in only 5 out of 14 years.</li>"
            "<li><b>Win Rate Signal:</b> February (78.6%), October (71.4%), and April (69.2%) have "
            "the highest win rates in the dataset - months where Bitcoin closed positive more than "
            "2 out of every 3 years.</li>"
            "<li><b>Note on June:</b> Despite the popular 'Sell in May' narrative, June shows a "
            "+9.0% average return and a 61.5% win rate in this dataset - not a weak month historically.</li>"
            "</ul>"
        )
    else:
        st.error("data/btc_data.csv not found.")

# =============================================================================
# Tab 4: Model Performance
# =============================================================================
with tabs[3]:
    if df_results is not None:
        st.markdown('<div class="cc-section-title">Performance Leaderboard</div>', unsafe_allow_html=True)
        st.write("Evaluate MAE, RMSE, and MAPE across the 1-Day, 3-Day, and 7-Day forecasting horizons.")

        # Find best model per horizon by lowest MAE
        best = (
            df_results.loc[df_results.groupby("Horizon")["MAE"].idxmin()]
            .set_index("Horizon")["Model"]
            .to_dict()
        )

        # Custom leaderboard HTML table
        rows_html = ""
        for _, row in df_results.iterrows():
            horizon  = row["Horizon"]
            model    = row["Model"]
            mae      = row["MAE"]
            rmse     = row["RMSE"]
            mape     = row["MAPE (%)"]
            is_best  = best.get(horizon) == model
            badge    = '<span class="badge-best">BEST</span>' if is_best else ""
            if mape < 5:
                mape_color = "#4ade80"
            elif mape < 10:
                mape_color = "#fb923c"
            else:
                mape_color = "#f87171"
            rows_html += (
                f'<div class="leaderboard-row">'
                f'<div style="width:80px"><span class="badge-horizon">{horizon}</span></div>'
                f'<div style="flex:1;color:#58a6ff;font-weight:500">{model}{badge}</div>'
                f'<div style="width:140px;text-align:right;font-weight:600;color:#e6edf3">${mae:,.2f}</div>'
                f'<div style="width:140px;text-align:right;color:#8b949e">${rmse:,.2f}</div>'
                f'<div style="width:120px;text-align:right;font-weight:600;color:{mape_color}">{mape:.2f}%</div>'
                f'</div>'
            )

        st.markdown(
            '<div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;overflow:hidden;margin-bottom:24px;">'
            '<div class="leaderboard-header">'
            '<div style="width:80px">Horizon</div>'
            '<div style="flex:1">Model Architecture</div>'
            '<div style="width:140px;text-align:right">MAE (USD)</div>'
            '<div style="width:140px;text-align:right">RMSE (USD)</div>'
            '<div style="width:120px;text-align:right">MAPE (%)</div>'
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

        callout(
            "Key Insight: Log Returns Eliminate Extrapolation Failure",
            "<p>Previous models trained on raw price suffered from <b>extrapolation failure</b> - "
            "the MinMaxScaler was fit on training prices (up to ~$63K), so any test price above "
            "$63K was clipped at 1.0, causing MAPE above 40%. "
            "The fix: train on stationary <b>log returns</b> r = ln(P[t+h] / P[t]), which are "
            "scale-invariant. Prices are reconstructed using P_hat = P[t] x exp(r_hat). "
            "This dropped 1D MAPE from 42.9% to 2.14% (RNN) and 7D MAPE from 48.2% to 6.09% (Transformer).</p>",
        )
    else:
        st.warning("model_comparison_results.csv not found. Run `python cryptocast.py` first.")

# =============================================================================
# Tab 5: Diagnostics
# =============================================================================
with tabs[4]:
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

# =============================================================================
# Tab 6: Macro & Halving Dynamics
# =============================================================================
with tabs[5]:
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
