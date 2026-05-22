# CryptoCast: Multi-Horizon Bitcoin Price Forecasting Using Deep Learning

A comprehensive deep learning project that compares **1D-CNN**, **RNN**, **LSTM**, and **Transformer** architectures for multi-horizon Bitcoin price prediction (1-day, 3-day, and 7-day ahead).

## Project Overview

| Aspect | Details |
|--------|---------|
| **Models** | 1D-CNN, RNN, LSTM, Transformer |
| **Forecast Horizons** | 1-Day (1D), 3-Day (3D), 7-Day (7D) |
| **Data** | Bitcoin Historical Data (2010-2024, ~5,000 daily records) |
| **Features** | Price, Open, High, Low, Volume |
| **Sequence Length** | 60 days (sliding window) |
| **Train/Test Split** | 80/20 (chronological, no shuffle) |

## Best Results

| Horizon | Best Model | MAE (USD) | RMSE (USD) | MAPE (%) |
|---------|------------|-----------|------------|----------|
| 1D | 1D-CNN | 4,595 | 6,201 | 12.92 |
| 3D | LSTM | 3,576 | 4,373 | 11.66 |
| 7D | LSTM | 5,124 | 7,468 | 12.47 |

**Key Finding:** 1D-CNN excels at short-term (1D) prediction due to efficient local pattern extraction, while LSTM dominates at longer horizons (3D, 7D) thanks to its gated memory mechanism for long-range dependencies.

## Project Structure

```
CryptoCast/
├── cryptocast.py                    # Main entry point (full pipeline)
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
├── meta.json                        # Dataset metadata
├── model_comparison_results.csv     # Results summary table
├── results.json                     # Results in JSON format
├── data/
│   └── btc_data.csv                 # Cleaned Bitcoin price data
├── src/
│   ├── step1_eda.py                 # Data loading, cleaning, EDA, preprocessing
│   ├── step2_train.py               # Train all 12 model-horizon combinations
│   ├── step3_viz.py                 # Generate comparison visualizations
│   └── train_model.py               # Train a single model/horizon
├── results/
│   ├── 1D-CNN_1D.json               # Individual model results
│   ├── 1D-CNN_3D.json
│   ├── 1D-CNN_7D.json
│   ├── RNN_1D.json
│   ├── RNN_3D.json
│   ├── RNN_7D.json
│   ├── LSTM_1D.json
│   ├── LSTM_3D.json
│   ├── LSTM_7D.json
│   ├── Transformer_1D.json
│   ├── Transformer_3D.json
│   └── Transformer_7D.json
├── visualizations/
│   ├── 01_price_time_series.png     # EDA: Price history
│   ├── 02_volume_plot.png           # EDA: Trading volume
│   ├── 03_ohlc_plot.png             # EDA: OHLC comparison
│   ├── 04_price_distribution.png    # EDA: Price distribution
│   ├── 05_return_distribution.png   # EDA: Daily returns
│   ├── 06_seasonal_boxplots.png     # EDA: Seasonal patterns
│   ├── 07_correlation_heatmap.png   # EDA: Feature correlations
│   ├── 08_rolling_statistics.png    # EDA: Rolling stats
│   ├── 09_loss_curves_*.png         # Training loss curves (per horizon)
│   ├── 10_actual_vs_predicted_*.png  # Actual vs Predicted (per horizon)
│   ├── 11_error_distribution_*.png   # Error distributions (per horizon)
│   ├── 12_mae_comparison.png        # MAE bar charts
│   ├── 13_rmse_comparison.png       # RMSE bar charts
│   ├── 14_mape_comparison.png       # MAPE bar charts
│   ├── 15_horizon_comparison.png    # Multi-horizon grouped comparison
│   └── 16_scatter_actual_vs_pred.png # Scatter: actual vs predicted
└── CryptoCast_Project_Report.pdf    # Full project report
```

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/CryptoCast.git
cd CryptoCast
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data
Place your Bitcoin historical data CSV file in the `data/` folder as `btc_data.csv`. The CSV should contain columns: `Date`, `Price`, `Open`, `High`, `Low`, `Vol.`, `Change %`.

Alternatively, if your raw CSV uses the format from investing.com or similar sources, name it `Bitcoin Historical Data.csv` and place it in `data/`. The code will automatically parse and clean it.

### 4. Run the Pipeline

**Full pipeline (EDA + Training + Visualizations):**
```bash
python cryptocast.py
```

**Or run steps individually:**
```bash
# Step 1: Data loading, EDA, preprocessing
python src/step1_eda.py

# Step 2: Train all models
python src/step2_train.py

# Step 3: Generate visualizations
python src/step3_viz.py
```

**Train a single model:**
```bash
python src/train_model.py 1D-CNN 1D
python src/train_model.py LSTM 7D
```

## Model Architectures

### 1D-CNN
- 3 causal Conv1D layers (64, 64, 32 filters, kernel=3)
- GlobalAveragePooling1D + Dropout(0.2)
- Dense(64) + Dense(1)
- Best for short-term local pattern detection

### RNN (SimpleRNN)
- 2 SimpleRNN layers (64, 32 units)
- Dropout(0.2) between layers
- Dense(32) + Dense(1)
- Baseline recurrent model

### LSTM
- 3 stacked LSTM layers (128, 64, 32 units)
- Dropout(0.2) between layers
- Dense(64) + Dense(1)
- Best for longer horizons via gated memory

### Transformer
- 2 Transformer blocks with 4 attention heads (key_dim=64)
- Feed-forward dimension: 128
- Residual connections + LayerNorm
- GlobalAveragePooling1D + Dense(64) + Dense(1)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | MSE |
| Epochs | 50 (with early stopping, patience=8) |
| Batch Size | 64 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=4) |
| Validation Split | 15% of training data |
| Random Seed | 42 |

## Key Insights

1. **1D-CNN dominates at 1-day horizon** - Causal convolutions efficiently capture short-term momentum and mean-reversion patterns with minimal overfitting.

2. **LSTM excels at 3-day and 7-day horizons** - The forget/input/output gating mechanism enables effective long-range dependency modeling critical for extended forecast windows.

3. **Transformer underperforms** - With only ~5,000 training samples, the self-attention mechanism lacks sufficient data to learn effective attention patterns. Transformers are known to be data-hungry.

4. **SimpleRNN suffers from vanishing gradients** - Without gating mechanisms, SimpleRNN cannot maintain useful information over longer sequences, leading to poor performance across all horizons.

## Technologies

- **Python 3.9+**
- **TensorFlow / Keras** - Deep learning framework
- **NumPy / Pandas** - Data manipulation
- **Scikit-learn** - Preprocessing & metrics
- **Matplotlib / Seaborn** - Visualization
- **ReportLab** - PDF report generation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Bitcoin historical data sourced from Yahoo Finance / investing.com
- Inspired by research on deep learning applications in cryptocurrency price forecasting
