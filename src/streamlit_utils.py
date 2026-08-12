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

def card(title, value, detail=""):
    st.markdown(
        f"""
        <div class="cc-card">
            <h4>{title}</h4>
            <div class="cc-value">{value}</div>
            {"<div class='cc-detail'>" + detail + "</div>" if detail else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

def callout(title, text, type="info", warn=False):
    cls = "cc-callout warn" if (warn or type == "warn") else "cc-callout"
    st.markdown(
        f"""
        <div class="{cls}">
            <h4>{title}</h4>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_stakeholder_narrative(page_num, total_pages, title, simple_explanation, connection_story, key_takeaway):
    """
    Renders an executive presentation narrative banner at the top of every dashboard page,
    explaining the page in plain business language and connecting it to the overall workflow.
    """
    st.markdown(
        f'''
        <div style="background: linear-gradient(135deg, #161b22 0%, #0d1117 100%); border: 1px solid #30363d; border-left: 4px solid #4ade80; border-radius: 10px; padding: 18px 22px; margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #4ade80;">
                    Step {page_num} of {total_pages} · Executive Storyline & Business Impact
                </span>
                <span style="background: #21262d; border: 1px solid #30363d; color: #8b949e; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;">
                    Stakeholder View
                </span>
            </div>
            <div style="font-size: 14px; color: #e6edf3; font-weight: 500; line-height: 1.5; margin-bottom: 10px;">
                <b>💡 Plain English Goal</b>: {simple_explanation}
            </div>
            <div style="font-size: 13px; color: #8b949e; line-height: 1.5; margin-bottom: 10px; background: #0d1117; padding: 10px 14px; border-radius: 6px; border: 1px solid #21262d;">
                🔗 <b>Workflow Connection</b>: {connection_story}
            </div>
            <div style="font-size: 13px; color: #4ade80; font-weight: 600;">
                🎯 <b>Executive Takeaway</b>: {key_takeaway}
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

DARK_LAYOUT = dict(
    plot_bgcolor="#0d1117",
    paper_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", color="#8b949e"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", color="#8b949e"),
    margin=dict(t=30, b=30, l=10, r=10),
)
