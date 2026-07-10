"""
CryptoCast - Streamlit Presentation Dashboard
==============================================
An interactive, high-end presentation dashboard to showcase the Bitcoin forecasting
Capstone project for your live evaluation/viva.

Usage:
    streamlit run app.py
"""

import os
import json
import subprocess
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Configure Streamlit Page
st.set_page_config(
    page_title="CryptoCast: Bitcoin Price Forecasting",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'btc_data.csv')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
RESULTS_CSV = os.path.join(PROJECT_DIR, 'model_comparison_results.csv')
VIZ_DIR = os.path.join(PROJECT_DIR, 'visualizations')

# Custom CSS for rich aesthetics and dark slate headers
st.markdown("""
<style>
    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: #1a5276;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        color: #566573;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #f8f9f9;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2e86c1;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .viva-box {
        background-color: #fdf2e9;
        border-left: 5px solid #e67e22;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }
    .leakage-warning {
        background-color: #fdebd0;
        border-left: 5px solid #d35400;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load data
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
        return df
    return None

df_raw = load_data()

# Sidebar Setup
st.sidebar.markdown("# 🪙 CryptoCast Dashboard")
st.sidebar.markdown("### Multi-Horizon Price Forecasting Using Deep Learning")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigate Project",
    [
        "Project Overview",
        "Exploratory Data Analysis",
        "Walk-Forward Validation",
        "Model Comparisons",
        "Detailed Model Diagnostics",
        "Viva Prep & Mentor Notes"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Viva Presentation Mode**\n\n"
    "This dashboard is pre-loaded with your leak-free metrics and visualizations. "
    "Use it to present your results during your GUVI Live Evaluation!"
)

# ----------------- Tab 1: Project Overview -----------------
if menu == "Project Overview":
    st.markdown("<div class='main-title'>CryptoCast: Bitcoin Price Forecasting</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Comparing 1D-CNN, RNN, LSTM, & Transformer architectures across multiple horizons</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "<div class='metric-card'><h4>📊 Dataset Size</h4>"
            "<h3>4,990+ Daily Records</h3>"
            "<small>Bitcoin price history: 2010 to 2024</small></div>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            "<div class='metric-card'><h4>⏱️ Sequence Length</h4>"
            "<h3>60 Days</h3>"
            "<small>Sliding window history context</small></div>",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            "<div class='metric-card'><h4>🎯 Targets (Horizons)</h4>"
            "<h3>1-Day, 3-Day, 7-Day</h3>"
            "<small>Predicting future prices (USD)</small></div>",
            unsafe_allow_html=True
        )
        
    st.markdown("### 📌 Problem Statement")
    st.write(
        "Cryptocurrency markets are highly volatile and subject to complex non-linear dynamics. "
        "Standard financial indicators fail to capture long-range temporal dependencies. "
        "This project evaluates deep learning architectures to establish standard benchmarks for multi-horizon price forecasting."
    )
    
    st.markdown("### 🧱 Selected Model Architectures")
    arch_col1, arch_col2 = st.columns(2)
    with arch_col1:
        st.markdown("#### 1. 1D Convolutional Neural Network (1D-CNN)")
        st.write(
            "Extracts local patterns using causal 1D filters. It trains very quickly and acts as "
            "a learnable technical indicator for short-term trends."
        )
        st.markdown("#### 2. Stacked Recurrent Neural Network (SimpleRNN)")
        st.write(
            "A baseline recurrent architecture for time-series modeling. It learns simple sequential trends "
            "but struggles with vanishing gradients."
        )
    with arch_col2:
        st.markdown("#### 3. Long Short-Term Memory (LSTM)")
        st.write(
            "Employs input, output, and forget gates to maintain gradients over long temporal sequences. "
            "Designed to capture long-term context and dependencies."
        )
        st.markdown("#### 4. Time-Series Transformer")
        st.write(
            "Leverages multi-head self-attention mechanism to capture global context across all sequence elements. "
            "Requires large datasets for effective generalization."
        )

# ----------------- Tab 2: Exploratory Data Analysis -----------------
elif menu == "Exploratory Data Analysis":
    st.markdown("<h2>Exploratory Data Analysis (EDA)</h2>", unsafe_allow_html=True)
    st.write("Analyze the statistical properties of historical Bitcoin prices and daily returns.")
    
    if df_raw is not None:
        st.markdown("### 📉 Interactive Historical Price Explorer")
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Price'], name='Close Price (USD)', line=dict(color='#1a5276', width=1.5)))
        fig_price.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Open'], name='Open Price (USD)', line=dict(color='#e67e22', width=1.0, dash='dash')))
        fig_price.update_layout(
            title="Bitcoin (BTC-USD) Closing and Opening Prices",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend_title="Legend",
            hovermode="x unified",
            height=450
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        st.markdown("### 🗃️ Cleaned Dataset Preview")
        st.dataframe(df_raw.tail(100), use_container_width=True)
        
        st.markdown("### 📊 Pre-rendered Statistical Visualizations")
        eda_images = {
            "Price Distribution": "04_price_distribution.png",
            "Return Distribution": "05_return_distribution.png",
            "Seasonal Boxplots": "06_seasonal_boxplots.png",
            "Correlation Heatmap": "07_correlation_heatmap.png",
            "Rolling Statistics": "08_rolling_statistics.png"
        }
        
        selected_img = st.selectbox("Select Visual Asset to inspect", list(eda_images.keys()))
        img_path = os.path.join(VIZ_DIR, eda_images[selected_img])
        if os.path.exists(img_path):
            st.image(img_path, caption=selected_img, use_container_width=True)
        else:
            st.warning(f"Visualization {img_path} not found. Run step1_eda.py to generate it.")
    else:
        st.error("Data file not found. Ensure `data/btc_data.csv` is generated.")

# ----------------- Tab 3: Walk-Forward Validation -----------------
elif menu == "Walk-Forward Validation":
    st.markdown("<h2>Walk-Forward Validation</h2>", unsafe_allow_html=True)
    st.write(
        "Walk-Forward (or Expanding Window) validation is the gold standard for time-series model evaluation. "
        "Instead of a static train/test split, the training window incrementally expands to simulate real-world trading retraining."
    )
    
    st.image(os.path.join(VIZ_DIR, "01_price_time_series.png"), caption="Bitcoin Time Series Context", use_container_width=True)
    
    st.markdown("### 📈 Run Walk-Forward Validation Demonstration")
    st.write("Click below to run a baseline walk-forward evaluation (train 450, test 90, expand by 90) on Bitcoin data.")
    
    if st.button("▶ Run Walk-Forward Demo"):
        demo_script = os.path.join(PROJECT_DIR, 'src', 'walk_forward_demo.py')
        if os.path.exists(demo_script):
            with st.spinner("Running walk-forward loops..."):
                res = subprocess.run([st.sidebar.text_input("Python command", value="python"), demo_script], capture_output=True, text=True)
                if res.returncode == 0:
                    st.success("Execution Complete!")
                    st.code(res.stdout, language="text")
                else:
                    st.error("Execution failed!")
                    st.code(res.stderr, language="text")
        else:
            st.error(f"Demo script {demo_script} not found.")
            
    st.markdown("""
    <div class='viva-box'>
        <h4>💡 Viva Insights: Why do we use Walk-Forward?</h4>
        <p>In standard machine learning, we shuffle datasets. In time-series, shuffling causes **look-ahead bias** (using future prices to predict past prices). 
        Walk-forward validation respects the temporal flow of time. Expanding the window (e.g. 450 -> 540 -> 630) allows the model to capture the latest market dynamics while preserving chronological integrity.</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------- Tab 4: Model Comparisons -----------------
elif menu == "Model Comparisons":
    st.markdown("<h2>Model Comparison & Performance Leaderboard</h2>", unsafe_allow_html=True)
    st.write("Compare model errors (MAE, RMSE, MAPE) across the 1D, 3D, and 7D forecast horizons.")
    
    if os.path.exists(RESULTS_CSV):
        df_results = pd.read_csv(RESULTS_CSV)
        st.markdown("### 🏆 Evaluation Metrics Table (Leak-Free)")
        st.dataframe(df_results, use_container_width=True)
        
        st.markdown("### 📊 Interactive Metrics Visualization")
        metric_opt = st.selectbox("Compare Metric", ["MAE", "RMSE", "MAPE (%)"])
        
        fig = px.bar(
            df_results, 
            x="Model", 
            y=metric_opt, 
            color="Horizon",
            barmode="group",
            title=f"Model Comparison based on {metric_opt}",
            color_discrete_sequence=["#3498db", "#2ecc71", "#9b59b6"]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Comparison metrics CSV not found. Please run training first to generate `model_comparison_results.csv`.")
        
    st.markdown("### 🖼️ Pre-rendered Comparison Visualizations")
    comp_images = {
        "MAE Comparison": "12_mae_comparison.png",
        "RMSE Comparison": "13_rmse_comparison.png",
        "MAPE Comparison": "14_mape_comparison.png",
        "Horizon-wise Comparison": "15_horizon_comparison.png"
    }
    selected_comp = st.selectbox("Select Visual Asset", list(comp_images.keys()))
    img_path = os.path.join(VIZ_DIR, comp_images[selected_comp])
    if os.path.exists(img_path):
        st.image(img_path, caption=selected_comp, use_container_width=True)
    else:
        st.warning(f"File {img_path} not found.")

# ----------------- Tab 5: Detailed Model Diagnostics -----------------
elif menu == "Detailed Model Diagnostics":
    st.markdown("<h2>Detailed Model Diagnostics</h2>", unsafe_allow_html=True)
    st.write("Inspect individual model runs, train/val loss curves, and prediction alignments.")
    
    horz = st.selectbox("Select Horizon", ["1D", "3D", "7D"])
    mdl = st.selectbox("Select Model Architecture", ["1D-CNN", "RNN", "LSTM", "Transformer"])
    
    st.markdown(f"### Diagnostic Visualizations for {mdl} ({horz})")
    
    # Render interactive plot from JSON results if present
    json_path = os.path.join(RESULTS_DIR, f"{mdl}_{horz}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            metrics_data = json.load(f)
            
        st.markdown("#### Test Set Actual vs. Predicted Prices")
        y_test = metrics_data['y_test']
        y_pred = metrics_data['y_pred']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test, name="Actual Price", line=dict(color="#1a5276", width=1.5)))
        fig.add_trace(go.Scatter(y=y_pred, name="Predicted Price", line=dict(color="#e74c3c", width=1.5, dash='dash')))
        fig.update_layout(
            title=f"{mdl} {horz} Horizon Predictions on Test Set",
            xaxis_title="Time Steps (Test Window)",
            yaxis_title="Bitcoin Price (USD)",
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("MAE (USD)", f"${metrics_data['MAE']:,.2f}")
        col_m2.metric("RMSE (USD)", f"${metrics_data['RMSE']:,.2f}")
        col_m3.metric("MAPE", f"{metrics_data['MAPE']:.2f}%")
        
        st.markdown("#### Training & Validation Loss Curves")
        loss = metrics_data['history']['loss']
        val_loss = metrics_data['history']['val_loss']
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=loss, name="Train Loss (MSE)", line=dict(color="#3498db")))
        fig_loss.add_trace(go.Scatter(y=val_loss, name="Val Loss (MSE)", line=dict(color="#e67e22", dash='dash')))
        fig_loss.update_layout(
            title=f"{mdl} {horz} Loss Curve",
            xaxis_title="Epochs",
            yaxis_title="MSE Loss",
            height=350
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    else:
        st.info("Interactive predictions not loaded. Showing pre-rendered visualizations instead. Run training to generate results JSONs.")
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            loss_img = os.path.join(VIZ_DIR, f"09_loss_curves_{horz}.png")
            if os.path.exists(loss_img):
                st.image(loss_img, caption=f"Loss Curves - {horz}", use_container_width=True)
        with col_img2:
            pred_img = os.path.join(VIZ_DIR, f"10_actual_vs_predicted_{horz}.png")
            if os.path.exists(pred_img):
                st.image(pred_img, caption=f"Actual vs Predicted - {horz}", use_container_width=True)

# ----------------- Tab 6: Viva Prep & Mentor Notes -----------------
elif menu == "Viva Prep & Mentor Notes":
    st.markdown("<h2>🎓 Data Science Viva Preparation Guide</h2>", unsafe_allow_html=True)
    st.write("An interactive revision tool designed to help you confidently answer examiner questions.")
    
    st.markdown("""
    <div class='leakage-warning'>
        <h4>🚨 Viva Question 1: What is Data Leakage in Preprocessing?</h4>
        <p><b>The Issue:</b> If you fit a MinMaxScaler on the *entire dataset* before splitting, the scaler uses the overall maximum and minimum (which are located in the future test data) to scale the training set. This is severe data leakage because the training weights learn scaled coordinates influenced by the test set.</p>
        <p><b>How We Fixed It:</b> We split the chronological dataset into train (80%) and test (20%). The MinMaxScaler was fitted **strictly on the training data**. The test data was then scaled using those training parameters (meaning some test values might scale below 0 or above 1, which is correct and normal).</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='viva-box'>
        <h4>🧠 Viva Question 2: Compare the Strengths of the 4 Architectures.</h4>
        <ul>
            <li><b>1D-CNN:</b> Uses causal filters to scan temporal patterns. Extremely fast to train, highly effective for short-term (1D) momentum detection.</li>
            <li><b>RNN (Simple):</b> Standard recurrent baseline. Suffers from vanishing gradients over our 60-day sequence, hence it fails to learn long-term dependencies.</li>
            <li><b>LSTM:</b> Features cell memory gates (forget, input, output). Reduces vanishing gradients, making it the best model for capturing longer-term trends (3D and 7D).</li>
            <li><b>Transformer:</b> Uses multi-head self-attention. Excellent at looking at the whole sequence simultaneously to locate dependencies. However, it is very data-hungry and underperforms on our dataset (~5,000 samples) because it overfits.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='viva-box'>
        <h4>📈 Viva Question 3: Explain the 3 Metrics.</h4>
        <ul>
            <li><b>MAE (Mean Absolute Error):</b> Calculates the average magnitude of absolute errors. Very easy to interpret in currency (USD) but doesn't penalize outliers.</li>
            <li><b>RMSE (Root Mean Squared Error):</b> Squares errors before averaging, then takes the square root. Heavily penalizes large errors, showing if the model suffers from massive misses.</li>
            <li><b>MAPE (Mean Absolute Percentage Error):</b> Expresses error as a percentage of actual prices. Scale-independent and very useful for comparison across different price regimes.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🖨️ Compile Project Report PDF")
    st.write("Click below to compile your final Capstone Project Report incorporating your latest results.")
    
    if st.button("📄 Generate PDF Report"):
        report_script = os.path.join(PROJECT_DIR, 'src', 'generate_pdf_report.py')
        if os.path.exists(report_script):
            with st.spinner("Generating PDF report..."):
                res = subprocess.run([st.sidebar.text_input("Python command (PDF)", value="python"), report_script], capture_output=True, text=True)
                if res.returncode == 0:
                    st.success(f"Report compiled successfully! Location: {OUTPUT_PDF}")
                else:
                    st.error("Report compilation failed!")
                    st.code(res.stderr, language="text")
        else:
            st.error(f"Report compiler script {report_script} not found.")
