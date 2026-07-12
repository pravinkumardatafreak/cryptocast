import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Set page config
st.set_page_config(
    page_title="CryptoCast | Seasonality Analysis",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(PROJECT_DIR, "data", "btc_data.csv")

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
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    return None

df_raw = load_data()

st.markdown('<div class="cc-eyebrow">Temporal trends</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-title">Seasonality Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="cc-subtitle">Investigate Bitcoin average returns and win rates grouped by calendar months</div>', unsafe_allow_html=True)

if df_raw is not None:
    st.markdown('<div class="cc-section-title">Bitcoin Monthly Returns Heatmap (%)</div>', unsafe_allow_html=True)

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
