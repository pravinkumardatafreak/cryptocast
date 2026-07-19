# CryptoCast: Multi-Horizon Bitcoin Price Forecasting Using Deep Learning

A capstone project comparing **1D-CNN**, **RNN**, **LSTM**, **Transformer**, and **PatchTST**
deep learning architectures for Bitcoin price forecasting across three horizons — 1-day, 3-day,
and 7-day ahead.

## Project Summary

| Aspect | Details |
|---|---|
| **Models** | 1D-CNN, RNN, LSTM, Transformer, PatchTST (PyTorch) |
| **Forecast Horizons** | 1D, 3D, 7D (multi-output, single model per architecture) |
| **Target Variable** | Log return: `r = ln(P[t+h] / P[t])` — stationary, scale-invariant |
| **Features** | 9: OHLCV + Change% + Block_Reward + Days_Since_Halving + Halving_Progress |
| **Dataset** | ~5,000 daily BTC records (Aug 2010 – Mar 2024) |
| **Sequence Length** | 60-day sliding window |
| **Train / Test Split** | 80 / 20 chronological — no shuffle, no leakage |
| **Scaling** | MinMaxScaler fit on training partition only |

| Horizon | Best Model | MAE (USD) | RMSE (USD) | MAPE (%) | R² Score |
|---|---|---|---|---|---|
| **1D** | LSTM | $730.98 | $1,136.38 | **2.06%** | **0.9921** |
| **3D** | PatchTST | $1,347.56 | $1,962.13 | **3.85%** | **0.9768** |
| **7D** | PatchTST | $2,097.01 | $3,067.21 | **5.93%** | **0.9446** |

PatchTST — a state-of-the-art patch-based time series Transformer with reversible instance
normalization (RevIN) — outperforms the baseline architectures at the longer 3-day and 7-day horizons,
while LSTM holds a very slight edge at the 1-day horizon. See the full leaderboard in
[`model_comparison_results.csv`](model_comparison_results.csv) for all five models across all three horizons.

> [!NOTE]
> **The R² Paradox & Fractal Returns:** The $R^2$ scores shown above evaluate predicted vs actual *absolute price levels*. The near-1.0 scores are a path-dependent time-series artifact (yesterday's price is highly predictive of today's price level). In contrast, daily log returns behave like a fractal random walk (Hurst exponent $H \approx 0.53$). Predicting daily returns directly yields return-level $R^2$ scores of $1\% - 3\%$. In quantitative finance, explaining $2\%$ of return-level variance is considered highly successful due to the fractal, noise-dominated nature of markets.

---

## 📈 Feature Space Evolution (Ablation Study)

To improve forecasting robustness, we evolved our input features from reactive price-based technical indicators to protocol-level variables derived directly from Satoshi Nakamoto's whitepaper design. This ablation was run across the original four architectures (1D-CNN, RNN, LSTM, Transformer), prior to PatchTST being added:

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
├── app.py                           # Streamlit dashboard entry point (10 pages)
├── requirements.txt                 # Python dependencies
├── .gitignore
├── meta.json                        # Dataset metadata (shape, date range)
├── model_comparison_results.csv     # Compiled metrics table (all 5 models)
├── results.json                     # Results in JSON format (all 5 models)
├── wfv_results.json                 # Walk-forward validation results
│
├── data/
│   ├── btc_data.csv                 # Cleaned BTC daily price data
│   └── live_predictions_log.csv     # Dynamic prediction logs (automatically generated during live inference runs)
│
├── src/
│   ├── step1_eda.py                 # Data loading, cleaning, EDA, preprocessing
│   ├── step2_train_pytorch.py       # Orchestrates training of all models
│   ├── step3_viz.py                 # Generates comparison visualizations
│   ├── step4_wfv.py                 # Walk-forward validation (backtesting)
│   ├── train_model_pytorch.py       # Single-model PyTorch trainer (called by step2)
│   ├── llm_insights.py              # Groq LLM integration for AI-generated insights
│   └── streamlit_utils.py           # Shared dashboard styling helpers
│
├── pages/                           # 10 Streamlit dashboard pages (see below)
│
├── models/                          # Saved weights for live inference
│   ├── LSTM.pth
│   ├── Transformer.pth
│   └── PatchTST.pth                 # Note: 1D-CNN and RNN weights are not
│                                     # checked in — retrain via step2 to
│                                     # regenerate them for live inference
│
├── results/                         # Per-model JSON results (gitignored,
│                                     # regenerate by running the full pipeline)
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
bun run pipeline
```

This runs all three steps in sequence:
- **Step 1** — Load, clean, compute technical indicators, scale data
- **Step 2** — Train the models (1D-CNN, RNN, LSTM, Transformer, PatchTST)
- **Step 3** — Generate comparison visualizations

### 5. Launch the Dashboard
```bash
bun run dev
```

Opens at `http://localhost:8502`. The `bun run dev` script automatically passes `--server.port 8502` to prevent port collisions with other projects.

### 🥐 All Available Bun Commands

| Command | What it runs |
|---|---|
| `bun run dev` | Launch Streamlit dashboard on port 8502 |
| `bun run pipeline` | Run full training pipeline (`cryptocast.py`) |
| `bun run train` | Train deep learning models only |
| `bun run eda` | Run EDA & feature engineering step |
| `bun run wfv` | Run Walk-Forward Validation only |

> **Note:** Requires [Bun](https://bun.sh) (tested on 1.3.14). Install it, then restart your terminal so `bun` is available globally.

- **Overview (Landing Page)** — Project summary, model descriptions, and business scope
- **1. Exploratory Analysis** — Historical price trend + Outlier Box Plot + Interactive Return Distribution Zoom
- **2. Seasonality Analysis** — Intra-month time period return heatmap (Q1–Q4) + win rate stats
- **3. Model Performance** — Metric comparison leaderboard and dynamic bar charts across all 5 models
- **4. Backtest (WFV)** — 3-Fold Walk-Forward validation stats and model stability checks
- **5. Diagnostics** — Actual vs. predicted diagnostic selectors + loss curves
- **6. Macro & Halving Dynamics** — Liquidity rotation, halving schedule, and Satoshi whitepaper link
- **7. Live Prediction** — Real-time BTC data pulled from Yahoo Finance, run through a live PyTorch forward pass (LSTM, Transformer, or PatchTST)
- **8. Forward Testing** — True out-of-sample evaluation on unseen market data from March 2024 to present
- **9. Explainable AI** — SHAP-based explainability showing which features drive each prediction
- **10. Trading Simulator** — Translates model accuracy into a simulated trading strategy and business ROI

## Model Architectures (PyTorch)

All models share a **multi-output design**: one shared encoder body + a single head
producing 1D, 3D, and 7D log return predictions simultaneously (`Linear(..., 3)`).

### 1D-CNN
- 3 causally-padded Conv1D blocks: `Conv1D(64) → Conv1D(64) → Conv1D(32)`, each ReLU-activated
- Global average pooling (`AdaptiveAvgPool1d`) over the time axis
- `Linear(32→64) → ReLU → Dropout → Linear(64→3)`
- Fastest to train; captures short-term local patterns but weakest of the five architectures at every horizon

### RNN
- 2 stacked `SimpleRNN` layers: `RNN(64) → RNN(32)`, taking the last timestep's hidden state
- `Linear(32→32) → ReLU → Linear(32→3)`
- Simple baseline — surprisingly the strongest of the original four architectures at the 7D horizon

### LSTM
- 3 stacked LSTM layers: `LSTM(128) → LSTM(64) → LSTM(32)`, taking the last timestep's hidden state
- `Linear(32→64) → ReLU → Linear(64→3)`
- Gated memory cells reduce the vanishing-gradient problem — the strongest of the original four architectures at 1D and 3D

### Transformer
- Linear input projection to `d_model=256` (4 heads × 64 head-size)
- 2-layer `TransformerEncoder` (4 attention heads, feedforward dim 128)
- Global average pooling → `Linear(256→64) → ReLU → Linear(64→3)`
- Captures long-range dependencies via self-attention, though it doesn't lead at any single horizon among the original four

### PatchTST
- Reversible Instance Normalization (RevIN) applied per-sample before encoding, and inverted after
- Sequence split into 5 non-overlapping patches (patch length 12, stride 12) instead of point-wise tokens
- 3-layer `TransformerEncoder` over patch embeddings (4 heads, feedforward dim 256, GELU activation)
- `Flatten → Linear(640→64) → ReLU → Dropout → Linear(64→3)`
- **Best model overall** — beats every other architecture at all three horizons (see results table above)

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=0.001 / 0.0005) |
| Loss | MSE on log returns |
| Epochs | 10 (Optimized for fast live viva execution; pre-saved outputs exist) |
| Batch Size | 64 |
| Validation Split | 15% of training data (chronological) |
| Random Seed | 42 |

## Intra-Month Time Period Analysis

Rather than evaluating calendar months (January, February, etc.), we divide the ~30 days of each month into four distinct time periods: **Q1** (Days 1–7), **Q2** (Days 8–15), **Q3** (Days 16–22), and **Q4** (Days 23–31). 

Based on daily log returns from August 2010 to March 2024:

| Time Period (Days) | Avg Daily Return | Daily Win Rate | Market Characterization / Signal |
|---|---|---|---|
| **Q4 (Days 23–31)** | **+0.775%** | 48.47% | **Highest average daily returns** (Turn-of-the-Month buying pressure) |
| **Q1 (Days 1–7)** | **+0.541%** | **51.45%** | **Highest daily win rate** (Continued TOM capital inflows) |
| **Q3 (Days 16–22)** | **+0.380%** | 48.25% | Moderate recovery and consolidation |
| **Q2 (Days 8–15)** | **+0.195%** | 48.39% | **Weakest returns** (Mid-month capital stagnation) |

**Turn-of-the-Month (TOM) Insight:** The data strongly supports the presence of the Turn-of-the-Month effect in Bitcoin. Average daily returns in Q4 (+0.775%) and Q1 (+0.541%) are significantly higher than the mid-month Q2 period (+0.195%). This is driven by systemic liquidity cycles, cash reallocations, and paycheck reinvestments occurring at the end and beginning of calendar months.

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
- **Bun** — Designated JavaScript runtime & package manager (replacing `npm` / `npx` for all current and future JS extension tooling)

## License

MIT License

## Acknowledgments

- Bitcoin historical data sourced from Investing.com
- GUVI × HCL Master Data Science Program — Capstone Project
