"""
CryptoCast - Step 1: Data Loading, Cleaning & Exploratory Data Analysis
=======================================================================
Loads Bitcoin price data from CSV, cleans it, generates EDA visualizations,
and saves preprocessed data for model training.

Usage:
    python src/step1_eda.py
"""

import os, warnings, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
VIZ_DIR = os.path.join(PROJECT_DIR, 'visualizations')
OUTPUT_DIR = PROJECT_DIR
os.makedirs(VIZ_DIR, exist_ok=True)

SEQ_LENGTH = 60
HORIZONS = [1, 3, 7]
HORIZON_NAMES = ['1D', '3D', '7D']
TEST_RATIO = 0.2

# ── Plot Styling ──────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ── 1. Data Loading & Cleaning ───────────────────────────────
print("=" * 70)
print("CryptoCast: Multi-Horizon Bitcoin Price Forecasting")
print("=" * 70)

print("\n[Step 1] Loading Bitcoin price data from CSV...")
csv_path = os.path.join(DATA_DIR, 'btc_data.csv')

# Check if raw CSV exists, otherwise look for uploaded data
raw_csv = os.path.join(DATA_DIR, 'Bitcoin Historical Data.csv')
if os.path.exists(raw_csv):
    raw = pd.read_csv(raw_csv, encoding='utf-8-sig')
    raw['Date'] = pd.to_datetime(raw['Date'], format='%d-%m-%Y')
    raw = raw.sort_values('Date').reset_index(drop=True)

    def parse_price(val):
        if isinstance(val, str):
            return float(val.replace(',', ''))
        return float(val)

    def parse_volume(val):
        if isinstance(val, str):
            val = val.strip()
            if val.endswith('K'):
                return float(val[:-1].replace(',', '')) * 1e3
            elif val.endswith('M'):
                return float(val[:-1].replace(',', '')) * 1e6
            elif val.endswith('B'):
                return float(val[:-1].replace(',', '')) * 1e9
            else:
                return float(val.replace(',', ''))
        return float(val)

    def parse_change(val):
        if isinstance(val, str):
            return float(val.replace('%', '').replace(',', ''))
        return float(val)

    data = pd.DataFrame()
    data['Date'] = raw['Date']
    data['Price'] = raw['Price'].apply(parse_price)
    data['Open'] = raw['Open'].apply(parse_price)
    data['High'] = raw['High'].apply(parse_price)
    data['Low'] = raw['Low'].apply(parse_price)
    data['Vol.'] = raw['Vol.'].apply(parse_volume)
    data['Change %'] = raw['Change %'].apply(parse_change)
    data = data.dropna().reset_index(drop=True)
    data = data.set_index('Date')
elif os.path.exists(csv_path):
    data = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
else:
    raise FileNotFoundError(
        "No data file found. Place 'Bitcoin Historical Data.csv' or 'btc_data.csv' in the data/ folder."
    )

print(f"  Dataset shape: {data.shape}")
print(f"  Date range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
print(f"  Price range: ${data['Price'].min():,.2f} - ${data['Price'].max():,.2f}")

# Save cleaned data
data.to_csv(csv_path)

# ── 2. EDA Visualizations ────────────────────────────────────
print("\n[Step 1] Generating EDA visualizations...")

# 01 - Price Time Series
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(data.index, data['Price'], color='#1a5276', linewidth=1.0, label='BTC Closing Price')
ax.set_title('Bitcoin (BTC-USD) Historical Closing Price', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12); ax.set_ylabel('Price (USD)', fontsize=12)
ax.legend(loc='upper left', fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '01_price_time_series.png'), bbox_inches='tight')
plt.close()
print("  [1/8] Price time series")

# 02 - Volume
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(data.index, data['Vol.']/1e9, color='#2e86c1', alpha=0.6, width=2)
ax.set_title('Bitcoin Daily Trading Volume', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12); ax.set_ylabel('Volume (Billion USD)', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '02_volume_plot.png'), bbox_inches='tight')
plt.close()
print("  [2/8] Volume plot")

# 03 - OHLC
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(data.index, data['Open'], alpha=0.5, label='Open', linewidth=0.7)
ax.plot(data.index, data['High'], alpha=0.5, label='High', linewidth=0.7)
ax.plot(data.index, data['Low'], alpha=0.5, label='Low', linewidth=0.7)
ax.plot(data.index, data['Price'], alpha=1.0, label='Close (Price)', linewidth=1.0, color='#1a5276')
ax.set_title('Bitcoin OHLC Price Comparison', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12); ax.set_ylabel('Price (USD)', fontsize=12)
ax.legend(loc='upper left', fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '03_ohlc_plot.png'), bbox_inches='tight')
plt.close()
print("  [3/8] OHLC plot")

# 04 - Price Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(data['Price'], bins=80, color='#2e86c1', alpha=0.7, edgecolor='white')
axes[0].set_title('Distribution of Bitcoin Closing Price', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Price (USD)', fontsize=12); axes[0].set_ylabel('Frequency', fontsize=12)
sns.kdeplot(data['Price'], ax=axes[1], color='#1a5276', linewidth=2)
axes[1].set_title('KDE of Bitcoin Closing Price', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Price (USD)', fontsize=12); axes[1].set_ylabel('Density', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '04_price_distribution.png'), bbox_inches='tight')
plt.close()
print("  [4/8] Price distribution")

# 05 - Return Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
returns = data['Change %'].dropna()
axes[0].hist(returns, bins=100, color='#e74c3c', alpha=0.7, edgecolor='white')
axes[0].set_title('Distribution of Daily Returns (%)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Daily Change (%)', fontsize=12); axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1)
stats.probplot(returns, plot=axes[1])
axes[1].set_title('Q-Q Plot of Daily Returns', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '05_return_distribution.png'), bbox_inches='tight')
plt.close()
print("  [5/8] Return distribution")

# 06 - Seasonal Boxplots
data_temp = data.copy()
data_temp['Year'] = data_temp.index.year
data_temp['Month'] = data_temp.index.month
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
yearly_data = [data_temp[data_temp['Year'] == y]['Price'].values for y in sorted(data_temp['Year'].unique())]
bp1 = axes[0].boxplot(yearly_data, tick_labels=sorted(data_temp['Year'].unique()), patch_artist=True)
for patch in bp1['boxes']: patch.set_facecolor('#aed6f1')
axes[0].set_title('Bitcoin Price by Year', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Year', fontsize=12); axes[0].set_ylabel('Price (USD)', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
monthly_data = [data_temp[data_temp['Month'] == m]['Price'].values for m in range(1, 13)]
bp2 = axes[1].boxplot(monthly_data, tick_labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], patch_artist=True)
for patch in bp2['boxes']: patch.set_facecolor('#a9dfbf')
axes[1].set_title('Bitcoin Price by Month', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Month', fontsize=12); axes[1].set_ylabel('Price (USD)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '06_seasonal_boxplots.png'), bbox_inches='tight')
plt.close()
print("  [6/8] Seasonal boxplots")

# 07 - Correlation Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
corr = data[['Open', 'High', 'Low', 'Price', 'Vol.', 'Change %']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.4f', cmap='Blues', vmin=-1, vmax=1, ax=ax, linewidths=0.5)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '07_correlation_heatmap.png'), bbox_inches='tight')
plt.close()
print("  [7/8] Correlation heatmap")

# 08 - Rolling Statistics
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
axes[0].plot(data.index, data['Price'], alpha=0.3, label='Price', linewidth=0.5)
axes[0].plot(data.index, data['Price'].rolling(30).mean(), label='30-Day MA', color='#e74c3c', linewidth=1.5)
axes[0].plot(data.index, data['Price'].rolling(90).mean(), label='90-Day MA', color='#2ecc71', linewidth=1.5)
axes[0].set_title('Bitcoin Price with Moving Averages', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Price (USD)', fontsize=12); axes[0].legend(loc='upper left', fontsize=11); axes[0].grid(True, alpha=0.3)
axes[1].plot(data.index, data['Price'].rolling(30).std(), label='30-Day Rolling Std', color='#e74c3c', linewidth=1.5)
axes[1].plot(data.index, data['Price'].rolling(90).std(), label='90-Day Rolling Std', color='#2ecc71', linewidth=1.5)
axes[1].set_title('Rolling Volatility (Standard Deviation)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Std Dev (USD)', fontsize=12); axes[1].set_xlabel('Date', fontsize=12)
axes[1].legend(loc='upper left', fontsize=11); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '08_rolling_statistics.png'), bbox_inches='tight')
plt.close()
print("  [8/8] Rolling statistics")

# ── 3. Preprocessing ─────────────────────────────────────────
print("\n[Step 1] Preprocessing data (leak-free & whitepaper protocol features)...")

# Whitepaper-derived features based on Bitcoin protocol hardcoded rules
def get_halving_features(dates):
    dates = pd.to_datetime(dates)
    h_dates = pd.to_datetime(['2009-01-03', '2012-11-28', '2016-07-09', '2020-05-11', '2024-04-19'])
    h_rewards = [50.0, 25.0, 12.5, 6.25, 3.125]
    
    rewards = []
    days_since = []
    progress = []
    
    for d in dates:
        past_idx = np.where(h_dates <= d)[0]
        if len(past_idx) == 0:
            epoch_idx = 0
        else:
            epoch_idx = past_idx[-1]
            
        reward = h_rewards[epoch_idx]
        last_h = h_dates[epoch_idx]
        
        diff_days = (d - last_h).days
        days_since.append(float(diff_days))
        rewards.append(reward)
        
        if epoch_idx < len(h_dates) - 1:
            next_h = h_dates[epoch_idx + 1]
            total_epoch_days = (next_h - last_h).days
            prog = diff_days / total_epoch_days
        else:
            next_h = pd.to_datetime('2028-04-17') # Next halving est.
            total_epoch_days = (next_h - last_h).days
            prog = diff_days / total_epoch_days
            
        progress.append(prog)
        
    return rewards, days_since, progress

rewards, days_since, progress = get_halving_features(data.index)
data['Block_Reward'] = rewards
data['Days_Since_Halving'] = days_since
data['Halving_Progress'] = progress

# Drop rows with NaNs (e.g. initial Change % row)
data = data.dropna()

# Save cleaned data with whitepaper features
data.to_csv(csv_path)

# Features list representing the updated whitepaper-derived dataset
features = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %', 'Block_Reward', 'Days_Since_Halving', 'Halving_Progress']
target_col = 'Price'

# Determine the chronological split index to avoid data leakage
split_idx = int(len(data) * (1 - TEST_RATIO))
print(f"  Training features split index: {split_idx} / {len(data)} samples")

# Fit the scalers strictly on the training partition
scaler = MinMaxScaler()
scaler.fit(data[features].iloc[:split_idx])

target_scaler = MinMaxScaler()
target_scaler.fit(data[[target_col]].iloc[:split_idx])

# Transform the entire dataset using the train-fitted scalers
scaled_data = scaler.transform(data[features])

np.save(os.path.join(OUTPUT_DIR, 'scaled_data.npy'), scaled_data)
with open(os.path.join(OUTPUT_DIR, 'scalers.pkl'), 'wb') as f:
    pickle.dump({'scaler': scaler, 'target_scaler': target_scaler}, f)

meta = {
    'seq_length': SEQ_LENGTH,
    'horizons': HORIZONS,
    'horizon_names': HORIZON_NAMES,
    'test_ratio': TEST_RATIO,
    'data_shape': list(data[features].shape),
    'date_range': [str(data.index.min()), str(data.index.max())],
    'data_source': 'Bitcoin Historical Data CSV',
    'num_samples': len(data)
}
with open(os.path.join(OUTPUT_DIR, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

for horizon, name in zip(HORIZONS, HORIZON_NAMES):
    n_sequences = len(scaled_data) - SEQ_LENGTH - horizon + 1
    split = int(n_sequences * (1 - TEST_RATIO))
    print(f"  {name}: {n_sequences} sequences, train={split}, test={n_sequences - split}")

print("\n[Step 1] Complete! Leak-free data ready for model training.")

