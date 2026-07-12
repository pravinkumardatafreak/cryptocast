# GUVI Capstone Project: CryptoCast - Multi-Horizon Bitcoin Price Forecasting

This project implements a robust deep learning pipeline in PyTorch to forecast Bitcoin log-returns across three forecasting horizons: **1-Day**, **3-Day**, and **7-Day** ahead.

---

## 1. Project Directory Structure

```
# GUVI Capstone Project: CryptoCast - Multi-Horizon Bitcoin Price Forecasting

This project implements a robust deep learning pipeline in PyTorch to forecast Bitcoin log-returns across three forecasting horizons: **1-Day**, **3-Day**, and **7-Day** ahead.

---

## 1. Project Directory Structure

```
|-- data/                      # Dataset and processed CSV files
|-- src/                       # Modulized pipeline source files
|   |-- step0_data_loader.py   # Loads raw data
|   |-- step1_preprocessing.py # Preprocesses and splits dataset
|   |-- step2_feature_eng.py   # Computes technical indicators
|   |-- step3_train.py         # Trains the PyTorch models
|   |-- step4_wfv.py           # Evaluates model stability using backtesting
|   |-- models.py              # Model architecture definitions (CNN, RNN, LSTM, Transformer)
|-- visualizations/            # Plotly generated figures and diagnostic charts
|-- app.py                     # Streamlit Gemini-style Dark Mode Dashboard
|-- cryptocast.py              # Orchestration entry point
|-- requirements.txt           # Dependencies
```

---

## 2. Key Data Science & Modeling Concepts (Viva Prep)

### A. Target Selection: Why Log-Returns over Raw Prices?
Predicting raw prices in time-series forecasting has a major flaw: **extrapolation failure**. Because prices trend upward/downward over time, the test set often contains prices far outside the training range. Deep learning models cannot extrapolate beyond what they have seen.

To solve this, we forecast **Log-Returns**:
$$r_t = \ln(P_{t+h}/P_t)$$

* **Stationarity**: Log-returns have a constant mean and variance, making them easier for deep models to learn.
* **Extrapolation**: Models only need to predict percentage changes, eliminating out-of-bounds prediction errors.

### B. Feature Engineering
We engineered 10  indicators, including trend (SMA_7, SMA_30), momentum (RSI_14), and volatility (Vol_30) features.

### C. Architecture Design (Shared Encoder + Horizon Heads)
Instead of training 3 separate models per architecture, we employ a **Multi-Output Shared Encoder**:
* A single sequence encoder (e.g., LSTM, Transformer) processes the historical 30-day sequence.
* Three distinct output heads (Linear layers) branch off to predict 1D, 3D, and 7D horizons. This allows the model to learn shared temporal features.

---


## 3. Backtesting Model Results (Model Comparison)

The models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE %).

| Horizon | Model | MAE | RMSE | MAPE (%) |
|---|---|---|---|---|
| **1D** | 1D-CNN | 1194.87 | 1960.04 | 3.22% |
| **1D** | RNN | **771.68** | **1199.04** | **2.14%** (Best) |
| **1D** | LSTM | 909.74 | 1318.10 | 2.52% |
| **1D** | Transformer | 1481.91 | 2248.43 | 3.82% |
|---|---|---|---|---|
| **3D** | 1D-CNN | 2998.99 | 4928.41 | 8.01% |
| **3D** | RNN | **1533.31** | **2221.77** | **4.26%** (Best) |
| **3D** | LSTM | 2008.46 | 2697.86 | 5.59% |
| **3D** | Transformer | 1773.47 | 2607.93 | 4.97% |
|---|---|---|---|---|
| **7D** | 1D-CNN | 2800.66 | 3844.68 | 7.68% |
| **7D** | RNN | 2658.14 | 3818.29 | 7.15% |
| **7D** | LSTM | 2952.46 | 4042.60 | 8.15% |
| **7D** | Transformer | **2219.87** | **3281.06** | **6.09%** (Best) |

---


## 4. Walk-Forward Validation (Backtesting)

Standard K-Fold Cross-Validation leaks future data into the past because it shuffles the data. Time series models must be evaluated using **Walk-Forward Validation**:
* We use a **3-Fold Expanding Window WFV**.
* The training window expands sequentially, and the model is evaluated on the subsequent chronological window.
* This ensures that training data always precedes validation data, preserving chronological order.
* Fold statistics show consistent stability, indicating that the PyTorch log-return architecture does not suffer from variance/extrapolation issues across different market cycles.

---

## 5. Streamlit Gemini Dark Mode Dashboard

We built a complete front-end dashboard (`app.py`) featuring:
1. **Interactive Price & Forecasting Charts**: Allows the user to select the forecasting horizon and model to view predictions.
2. **Seasonality Heatmaps**: Analyzes win-rates by day-of-week and month-of-year, proving statistical seasonal anomalies in Bitcoin's history.
3. **Macro Indicators & Halving Cycles**: Educates investors on halving milestones (2012, 2016, 2020, 2024) and their historical correlation with price peaks.
 with price peaks.

 ---

 ## 6. How to Run the Project

 1. **Install dependencies**: `pip install -r requirements.txt`
 2. **Run complete pipeline**: `python cryptocast.py`
 3. **Launch dashboard**: `streamlit run app.py`

 This project serves as a comprehensive time-series forecasting presentation for GUVI Capstone Evaluation.
 
