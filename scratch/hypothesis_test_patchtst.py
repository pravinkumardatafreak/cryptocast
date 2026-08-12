import os
import json
import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = os.path.join('data', 'btc_data.csv')
RESULTS_PATH = os.path.join('results', 'PatchTST_7D.json')

def run_patchtst_hypothesis_test():
    """
    Evaluates the combined hypothesis:
    H0: Returns on (PatchTST 7D Bullish + Mon/Tue + Weekly Open Discount) days = Baseline Returns.
    H1: Returns on (PatchTST 7D Bullish + Mon/Tue + Weekly Open Discount) days > Baseline Returns.
    """
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    # Load PatchTST 7D test prediction results
    if not os.path.exists(RESULTS_PATH):
        print("PatchTST_7D.json not found.")
        return

    with open(RESULTS_PATH, 'r') as f:
        res = json.load(f)

    y_test = np.array(res['y_test'])
    y_pred = np.array(res['y_pred'])

    # Test set length
    test_len = len(y_test)
    test_df = df.iloc[-test_len:].copy()
    test_df['PatchTST_7D_Pred'] = y_pred
    test_df['Actual_7D_Price'] = y_test

    # Calculate 7D forward percentage return: (Actual_7D - Price) / Price * 100
    test_df['Forward_7D_Return'] = (test_df['Actual_7D_Price'] - test_df['Price']) / test_df['Price'] * 100

    # Calculate Weekly Open
    week_opens = []
    for idx_dt in test_df.index:
        monday_dt = idx_dt - pd.Timedelta(days=idx_dt.dayofweek)
        match = df[df.index >= monday_dt]
        if not match.empty:
            week_opens.append(match['Open'].iloc[0])
        else:
            week_opens.append(test_df.loc[idx_dt, 'Open'])
    test_df['Week_Open'] = week_opens

    # Combined Strategy Mask:
    # 1. PatchTST 7D predicts higher price than current price
    # 2. PatchTST 7D predicts higher price than Weekly Open (Anticipated Bullish Week)
    # 3. Day of week is Monday (0) or Tuesday (1)
    # 4. Current price is below Weekly Open
    patchtst_bullish = test_df['PatchTST_7D_Pred'] > test_df['Price']
    anticipated_bullish = test_df['PatchTST_7D_Pred'] > test_df['Week_Open']
    mon_tue = test_df.index.dayofweek.isin([0, 1])
    below_open = test_df['Price'] < test_df['Week_Open']

    combined_mask = patchtst_bullish & anticipated_bullish & mon_tue & below_open

    combined_returns = test_df.loc[combined_mask, 'Forward_7D_Return'].dropna()
    baseline_returns = test_df.loc[~combined_mask, 'Forward_7D_Return'].dropna()

    print("======================================================================")
    print("  PATCHTST AI + MON/TUE WEEKLY DISCOUNT HYPOTHESIS TEST RESULTS      ")
    print("======================================================================")
    print(f"Test Set Period Samples Analyzed           : {len(test_df):,}")
    print(f"Combined AI Strategy Filter Entry Days     : {len(combined_returns):,} samples")
    print(f"Baseline Days                              : {len(baseline_returns):,} samples\n")

    mean_strat = combined_returns.mean() if len(combined_returns) > 0 else 0
    std_strat = combined_returns.std() if len(combined_returns) > 0 else 0
    mean_base = baseline_returns.mean()
    std_base = baseline_returns.std()

    print(f"AI Strategy Mean 7D Return  : +{mean_strat:.4f}% (Std: {std_strat:.2f}%)")
    print(f"Baseline Mean 7D Return     : +{mean_base:.4f}% (Std: {std_base:.2f}%)\n")

    if len(combined_returns) > 2:
        t_stat, p_val_ttest = stats.ttest_ind(combined_returns, baseline_returns, equal_var=False, alternative='greater')
        u_stat, p_val_mwu = stats.mannwhitneyu(combined_returns, baseline_returns, alternative='greater')

        print("----------------------------------------------------------------------")
        print("1. Two-Sample Welch's t-Test Results:")
        print(f"   t-Statistic         = {t_stat:.4f}")
        print(f"   p-Value (One-Tailed) = {p_val_ttest:.6f}")

        print("\n2. Mann-Whitney U Test Results (Non-Parametric):")
        print(f"   U-Statistic         = {u_stat:,.1f}")
        print(f"   p-Value (One-Tailed) = {p_val_mwu:.6f}")

        print("----------------------------------------------------------------------")
        print("DECISION & CONCLUSION:")
        if p_val_ttest < 0.05:
            print("  [ACCEPT H1] REJECT H0: Combined PatchTST AI + Weekly Discount achieves STATISTICALLY SIGNIFICANT excess returns (p < 0.05).")
        else:
            print("  [FAIL TO REJECT H0]: p-value >= 0.05.")

if __name__ == '__main__':
    run_patchtst_hypothesis_test()
