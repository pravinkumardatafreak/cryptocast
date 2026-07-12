# CryptoCast: Multi-Horizon Bitcoin Price Forecasting Using Deep Learning

A capstone project comparing **1D-CNN**, **RNN**, **LSTM**, and **Transformer** deep learning
architectures for Bitcoin price forecasting across three horizons — 1-day, 3-day, and 7-day ahead.

## Project Summary

| Aspect | Details |
|---|---|
| **Models** | 1D-CNN, RNN, LSTM, Transformer (PyTorch) |
| **Forecast Horizons** | 1D, 3D, 7D (multi-output, single model per architecture) |
| **Target Variable** | Log return: `r = ln(P[t+h] / P[t])` — stationary, scale-invariant |
| **Features** | 9: OHLCV + Change% + Block_Reward + Days_Since_Halving + Halving_Progress |
| **Dataset** | ~5,000 daily BTC records (Aug 2010 – Mar 2024) |
| **Sequence Length** | 60-day sliding window |
| **Train / Test Split** | 80 / 20 chronological — no shuffle, no leakage |
| **Scaling** | MinMaxScaler fit on training partition only |

## Best Results (Leak-Free, Log-Return Architecture)

| Horizon | Best Model | MAE (USD) | RMSE (USD) | MAPE (%) |
|---|---|---|---|---|
| **1D** | LSTM | $753.74 | $1,169.24 | **2.12%** |
| **3D** | LSTM | $1,514.05 | $2,191.28 | **4.20%** |
| **7D** | Transformer | $2,127.22 | $3,060.78 | **6.04%** |

---

## 📈 Feature Space Evolution (Ablation Study)

To improve forecasting robustness, we evolved our input features from reactive price-based technical indicators to protocol-level variables derived directly from Satoshi Nakamoto's whitepaper design:

* **Version 1 (Price-Based Indicators):** `[OHLCV, Change %, SMA_7, SMA_30, RSI_14, Vol_30]`
* **Version 2 (Whitepaper Protocol Features):** `[OHLCV, Change %, Block_Reward, Days_Since_Halving, Halving_Progress]`

### Performance Impact of Feature Evolution (Best MAPEs)

| Horizon | Version 1 (Technical Indicators) | Version 2 (Whitepaper Features) | Performance Change |
|---|---|---|---|
| **1D** | 2.14% (RNN) | **2.12% (LSTM)** | **-0.02%** (LSTM takes the lead) |
| **3D** | 4.26% (RNN) | **4.20% (LSTM)** | **-0.06%** (RNN beaten by LSTM) |
| **7D** | 6.09% (Transformer) | **6.04% (Transformer)** | **-0.05%** (Transformer MAE drops by $92) |

**Key takeaway:** Replacing lagging, reactive indicators with deterministic whitepaper features successfully anchors the sequence models (LSTM and Transformer) to Bitcoin's structural supply-side cycles, reducing prediction error and increasing generalisation.

---

## ⚠️ Core Bottlenecks & Limitations of Forecasting

During evaluation, be prepared to discuss these three structural bottlenecks in Bitcoin price forecasting:

1. **The "Data Starvation" Bottleneck:**
   Deep learning models (especially Transformers) are highly data-hungry. In its entire 15-year history, Bitcoin has only completed **4 halving cycles**. The model has very few cycle transitions to learn from, limiting its ability to achieve asymptotic accuracy.
2. **The "2140 Protocol Boundary":**
   According to the Bitcoin whitepaper protocol, block reward halvings will terminate around the year **2140** when the total supply of 21 million BTC is reached. Post-2140, the `Block_Reward` drops permanently to `0`, meaning these cyclical features will become static.
3. **The "Simulation Trap" of Synthetic Data:**
   While we can easily project `Block_Reward` and `Halving_Progress` 1,000 years into the future with 100% certainty, we cannot project price. Training a model on synthetic future prices simply teaches it the mathematical rules of our generator script, failing to generalize to real human market psychology (fear, FOMO, regulations).


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
| Optimizer | Adam (lr=0.001 / 0.0005) |
| Loss | MSE on log returns |
| Epochs | 10 (Optimized for fast live viva execution; pre-saved outputs exist) |
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
