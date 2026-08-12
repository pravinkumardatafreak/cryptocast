import os
import numpy as np
import pandas as pd
import json

DATA_PATH = os.path.join('data', 'btc_data.csv')
RESULTS_PATH = os.path.join('results', 'PatchTST_7D.json')

def test_strategy_refinement():
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

    # Macro 50 SMA
    test_df['SMA_50'] = test_df['Price'].rolling(50, min_periods=1).mean()

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

    # Test Strategy Rules
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
        sma_50 = test_df['SMA_50'].iloc[t]

        is_risk_on = current_price >= sma_50
        patchtst_bullish = pred_7d_price > current_price
        anticipated_bullish = pred_7d_price > week_open
        is_mon_or_tue = dt.dayofweek in [0, 1]
        below_week_open = current_price < week_open

        # Refined Confluence: Risk-On + PatchTST Bullish + Mon/Tue + Weekly Open Discount
        buy_condition = is_risk_on and patchtst_bullish and anticipated_bullish and is_mon_or_tue and below_week_open
        sell_condition = (not is_risk_on) or (dt.dayofweek == 6) or (current_price >= pred_7d_price)

        portfolio_val = cash + (btc_held * current_price)
        equity_curve.append(portfolio_val)

        if buy_condition and not in_position:
            btc_held = portfolio_val / current_price
            cash = 0.0
            entry_price = current_price
            in_position = True
            trades += 1

        elif sell_condition and in_position:
            cash = btc_held * current_price
            btc_held = 0.0
            in_position = False
            if current_price > entry_price:
                wins += 1

    final_val = cash + (btc_held * test_df['Price'].iloc[-1])
    roi = ((final_val - initial_capital) / initial_capital) * 100
    win_rate = (wins / trades * 100) if trades > 0 else 0

    print("======================================================================")
    print("      REFINED PATCHTST MASTER STRATEGY SIMULATION RESULTS            ")
    print("======================================================================")
    print(f"Initial Capital   : ${initial_capital:,.2f}")
    print(f"Final Value       : ${final_val:,.2f}")
    print(f"Strategy ROI      : {roi:+.2f}%")
    print(f"Trades Executed   : {trades}")
    print(f"Win Rate          : {win_rate:.1f}%")

if __name__ == '__main__':
    test_strategy_refinement()
