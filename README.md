# CryptoCast: Multi-Horizon Bitcoin Price Forecasting Using Deep Learning

A capstone project comparing **1D-CNN**, **RNN**, **LSTM**, and **Transformer** deep learning
architectures for Bitcoin price forecasting across three horizons — 1-day, 3-day, and 7-day ahead.

## Project Summary

| Aspect | Details |
|---|---|
| **Models** | 1D-CNN, RNN, LSTM, Transformer (PyTorch) |
| **Forecast Horizons** | 1D, 3D, 7D (multi-output, single model per architecture) |
| **Target Variable** | Log return: `r = ln(P[t+h] / P[t])` — stationary, scale-invariant |
| **Features** | 10: OHLCV + Change% + SMA_7 + SMA_30 + RSI_14 + Vol_30 |
| **Dataset** | ~5,000 daily BTC records (Aug 2010 – Mar 2024) |
| **Sequence Length** | 60-day sliding window |
| **Train / Test Split** | 80 / 20 chronological — no shuffle, no leakage |
| **Scaling** | MinMaxScaler fit on training partition only |

## Best Results (Leak-Free, Log-Return Architecture)

| Horizon | Best Model | MAE (USD) | RMSE (USD) | MAPE (%) |
|---|---|---|---|---|
| **1D** | RNN | $771.68 | $1,199.04 | **2.14%** |
| **3D** | RNN | $1,533.31 | $2,221.77 | **4.26%** |
| **7D** | Transformer | $2,219.87 | $3,281.06 | **6.09%** |

**Key methodology decision:** Training on log returns (instead of raw price) eliminated
extrapolation failure — 1D MAPE dropped from 42.9% → 2.14% after the switch.
Price is reconstructed as `P_hat = P[t] × exp(r_hat)`.

## Project Structure

```
cryptocast/
├── cryptocast.py                    # Main entry point — runs full pipeline
├── app.py                           # Streamlit dashboard (6 tabs)
├── requirements.txt                 # Python dependencies
├── .gitignore
├── meta.json                        # Dataset metadata (shape, date range)
├── model_comparison_results.csv     # Compiled metrics table
├── results.json                     # Results in JSON format
│
├── data/
│   └── btc_data.csv                 # Cleaned BTC daily price data
│
├── src/
│   ├── step1_eda.py                 # Data loading, cleaning, EDA, preprocessing
│   ├── step2_train_pytorch.py       # Orchestrates training of all 4 models
│   ├── step3_viz.py                 # Generates comparison visualizations
│   └── train_model_pytorch.py       # Single-model PyTorch trainer (called by step2)
│
├── results/                         # Per-model JSON results (gitignored)
│   ├── 1D-CNN_1D.json
│   └── ...
│
└── visualizations/
    ├── 01_price_time_series.png
    ├── 02_volume_plot.png
    ├── 03_ohlc_plot.png
    ├── 04_price_distribution.png
    ├── 05_return_distribution.png
    ├── 06_seasonal_boxplots.png
    ├── 07_correlation_heatmap.png
    ├── 08_rolling_statistics.png
    ├── 09_loss_curves_{1D/3D/7D}.png
    ├── 10_actual_vs_predicted_{1D/3D/7D}.png
    └── ...
```

## Setup & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/pravinkumardatafreak/cryptocast.git
cd cryptocast
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 3. Prepare Data
Place your Bitcoin historical data CSV in `data/` as either:
- `Bitcoin Historical Data.csv` — raw Investing.com format (auto-parsed)
- `btc_data.csv` — pre-cleaned format

Required columns: `Date, Price, Open, High, Low, Vol., Change %`

### 4. Run the Full Pipeline
```bash
python cryptocast.py
```

This runs all three steps in sequence:
- **Step 1** — Load, clean, compute technical indicators, scale data
- **Step 2** — Train all 4 models (1D-CNN, RNN, LSTM, Transformer)
- **Step 3** — Generate comparison visualizations

### 5. Launch the Dashboard
```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with 6 tabs:
1. **Overview** — Project summary and model descriptions
2. **Exploratory Analysis** — Interactive price chart + 5 static EDA plots
3. **Seasonality Analysis** — Monthly return heatmap + win rate analysis
4. **Model Performance** — Leaderboard, metric comparison bar charts
5. **Diagnostics** — Per-model actual vs. predicted + loss curves
6. **Macro & Halving Dynamics** — Risk-on/off, halving cycle, Bitcoin whitepaper

## Model Architectures (PyTorch)

All models share a **multi-output design**: one shared encoder body + three parallel
output heads predicting 1D, 3D, and 7D log returns simultaneously.

### 1D-CNN
- Conv1D(64, kernel=3) → ReLU → MaxPool → Conv1D(128, kernel=3) → ReLU → MaxPool
- Flatten → Linear(64) → Three output heads
- Best for short-term local pattern detection

### RNN
- SimpleRNN(64) → SimpleRNN(32) → Linear(16) → Three output heads
- Baseline recurrent model — best performer at 1D and 3D horizons

### LSTM
- LSTM(64) → LSTM(32) → Linear(16) → Three output heads
- Gated memory cells reduce vanishing gradient problem

### Transformer
- MultiheadAttention(4 heads, d_model=64) + FeedForward(128) × 2 blocks
- Global pooling → Linear(32) → Three output heads
- Best performer at 7D horizon (global context via self-attention)

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss | MSE on log returns |
| Epochs | 100 (EarlyStopping, patience=15) |
| Batch Size | 64 |
| Validation Split | 15% of training data (chronological) |
| Random Seed | 42 |

## Seasonality Analysis

Based on 163 monthly observations from the dataset (Aug 2010 – Mar 2024):

| Month | Avg Return | Win Rate | Signal |
|---|---|---|---|
| April | +38.3% | 69.2% | Strong Buy |
| November | +38.7% | 57.1% | Strong Buy |
| October | +21.0% | 71.4% | High confidence |
| February | +17.6% | 78.6% | Highest win rate |
| September | -4.8% | 35.7% | Historically weakest |
| August | -0.1% | 38.5% | Marginally negative |

**Note:** June averages +9.0% with 61.5% win rate — the popular "Sell in May" narrative
does not hold in this dataset.

## Macroeconomic Context

### Liquidity Rotation
- **Risk-On** (low rates, QE): Capital flows into BTC — bull market tailwind
- **Risk-Off** (rate hikes, QT): Capital exits BTC — bear market trigger

### Bitcoin Halving Cycle
Every 210,000 blocks (~4 years), block rewards halve — creating a supply-side shock:

| Event | Date | Reward Before | Reward After |
|---|---|---|---|
| 1st Halving | Nov 28, 2012 | 50 BTC | 25 BTC |
| 2nd Halving | Jul 9, 2016 | 25 BTC | 12.5 BTC |
| 3rd Halving | May 11, 2020 | 12.5 BTC | 6.25 BTC |
| 4th Halving | Apr 19, 2024 | 6.25 BTC | 3.125 BTC |

Reference: [Bitcoin Whitepaper — Satoshi Nakamoto](https://bitcoin.org/bitcoin.pdf)

## Tech Stack

- **PyTorch** — Deep learning framework
- **Streamlit** — Interactive dashboard
- **Pandas / NumPy** — Data manipulation
- **Scikit-learn** — Preprocessing & metrics
- **Plotly** — Interactive charts
- **Matplotlib / Seaborn** — Static visualizations

## License

MIT License

## Acknowledgments

- Bitcoin historical data sourced from Investing.com
- GUVI × HCL Master Data Science Program — Capstone Project
