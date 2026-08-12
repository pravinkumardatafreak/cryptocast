import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import pandas as pd
from src.macro_regime import classify_3state_macro_regimes

DATA_PATH = os.path.join('data', 'btc_data.csv')
RESULTS_PATH = os.path.join('results', 'PatchTST_7D.json')

def test_3state_strategy():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    with open(RESULTS_PATH, 'r') as f:
        res = json.load(f)

    y_test = np.array(res['y_test'])
    y_pred = np.array(res['y_pred'])

    test_len = len(y_test)
    test_df = df.iloc[-test_len:].copy()
    test_df['PatchTST_7D_Pred'] = y_pred

    # Classify 3-State Regimes
    test_df = classify_3state_macro_regimes(test_df)

    # Weekly Open
    week_opens = []
    for idx_dt in test_df.index:
        monday_dt = idx_dt - pd.Timedelta(days=idx_dt.dayofweek)
        match = df[df.index >= monday_dt]
        if not match.empty:
            week_opens.append(match['Open'].iloc[0])
        else:
            week_opens.append(test_df.loc[idx_dt, 'Open'])
    test_df['Week_Open'] = week_opens

    initial_capital = 10000.0
    cash = initial_capital
    btc_held = 0.0
    in_position = False
    entry_price = 0.0
    trades = 0
    wins = 0

    equity_curve = []

    for t in range(len(test_df)):
        dt = test_df.index[t]
        current_price = test_df['Price'].iloc[t]
        pred_7d_price = test_df['PatchTST_7D_Pred'].iloc[t]
        week_open = test_df['Week_Open'].iloc[t]
        regime = test_df['Macro_Regime'].iloc[t]
        alloc_ratio = test_df['Regime_Allocation'].iloc[t]

        patchtst_bullish = pred_7d_price > current_price
        anticipated_bullish = pred_7d_price > week_open
        is_mon_or_tue = dt.dayofweek in [0, 1]
        below_week_open = current_price < week_open

        # ENTRY: Only trade in Bull or Consolidation regime (alloc_ratio > 0)
        buy_condition = (alloc_ratio > 0) and patchtst_bullish and anticipated_bullish and is_mon_or_tue and below_week_open
        sell_condition = (alloc_ratio == 0) or (dt.dayofweek == 6) or (pred_7d_price <= current_price)

        portfolio_val = cash + (btc_held * current_price)
        equity_curve.append(portfolio_val)

        if buy_condition and not in_position:
            trade_cash = portfolio_val * alloc_ratio
            btc_held = trade_cash / current_price
            cash = portfolio_val - trade_cash
            entry_price = current_price
            in_position = True
            trades += 1

        elif sell_condition and in_position:
            cash = cash + (btc_held * current_price)
            btc_held = 0.0
            in_position = False
            if current_price > entry_price:
                wins += 1

    final_val = cash + (btc_held * test_df['Price'].iloc[-1])
    roi = ((final_val - initial_capital) / initial_capital) * 100
    win_rate = (wins / trades * 100) if trades > 0 else 0

    regime_counts = test_df['Macro_Regime'].value_counts()

    print("======================================================================")
    print("      3-STATE MACRO REGIME (BULL, CONSOLIDATION, BEAR) RESULTS        ")
    print("======================================================================")
    print("Regime Distribution in Test Period:")
    for reg, cnt in regime_counts.items():
        print(f"  • {reg}: {cnt} days ({cnt/len(test_df)*100:.1f}%)")

    print(f"\nInitial Capital   : ${initial_capital:,.2f}")
    print(f"Final Value       : ${final_val:,.2f}")
    print(f"Strategy ROI      : {roi:+.2f}%")
    print(f"Trades Executed   : {trades}")
    print(f"Win Rate          : {win_rate:.1f}%")

if __name__ == '__main__':
    test_3state_strategy()
