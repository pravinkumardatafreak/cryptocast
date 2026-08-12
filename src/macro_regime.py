import numpy as np
import pandas as pd

def classify_3state_macro_regimes(df, sma_short=20, sma_long=50, band_pct=3.0):
    """
    Classifies market state into 3 distinct quantitative macro regimes:
      1. Bull Regime (Uptrend / Risk-On): Price > SMA50 and SMA20 > SMA50 -> 100% Capital Allocation.
      2. Consolidation Regime (Sideways / Range-Bound): |Price - SMA50| / SMA50 <= 3% -> 50% Allocation (Discount entries only).
      3. Bear Regime (Downtrend / Risk-Off): Price < SMA50 and SMA20 < SMA50 -> 0% Allocation (100% Cash Protection).

    Returns:
        pd.DataFrame with 'SMA_20', 'SMA_50', 'Band_Dist_Pct', 'Regime' columns.
    """
    data = df.copy()
    data['SMA_20'] = data['Price'].rolling(window=sma_short, min_periods=1).mean()
    data['SMA_50'] = data['Price'].rolling(window=sma_long, min_periods=1).mean()
    data['Band_Dist_Pct'] = (data['Price'] - data['SMA_50']) / data['SMA_50'] * 100.0

    regimes = []
    allocations = []

    for i in range(len(data)):
        p = data['Price'].iloc[i]
        s20 = data['SMA_20'].iloc[i]
        s50 = data['SMA_50'].iloc[i]
        dist = abs(data['Band_Dist_Pct'].iloc[i])

        # Consolidation Check: Price oscillating within +/- 3% of SMA_50
        if dist <= band_pct:
            regimes.append('Consolidation (Sideways Range)')
            allocations.append(0.50)
        # Bull Check: Price > SMA50 and SMA20 > SMA50
        elif p > s50 and s20 >= s50:
            regimes.append('Bull Regime (Uptrend)')
            allocations.append(1.00)
        # Bear Check: Price < SMA50 and SMA20 < SMA50
        elif p < s50 and s20 < s50:
            regimes.append('Bear Regime (Downtrend)')
            allocations.append(0.00)  # 100% Cash Protection!
        else:
            # Fallback range-bound
            regimes.append('Consolidation (Sideways Range)')
            allocations.append(0.50)

    data['Macro_Regime'] = regimes
    data['Regime_Allocation'] = allocations
    return data
