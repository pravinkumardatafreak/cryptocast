import os
import pandas as pd
import numpy as np
from scipy import stats

DATA_PATH = os.path.join('data', 'btc_data.csv')

def run_hypothesis_testing():
    """
    Formulates and executes rigorous Statistical Hypothesis Tests:
      H0 (Null Hypothesis): Average 7-day return following Monday/Tuesday Weekly Open Discount days 
                            is EQUAL to the baseline 7-day return.
      H1 (Alternative Hypothesis): Average 7-day return following Monday/Tuesday Weekly Open Discount days 
                                    is GREATER than the baseline 7-day return.
    """
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    # Calculate 7-day forward return: (Price_{t+7} - Price_t) / Price_t * 100
    df['Forward_7D_Return'] = (df['Price'].shift(-7) - df['Price']) / df['Price'] * 100

    # Calculate Weekly Open Price for each day
    week_opens = []
    for idx_dt in df.index:
        monday_dt = idx_dt - pd.Timedelta(days=idx_dt.dayofweek)
        match = df[df.index >= monday_dt]
        if not match.empty:
            week_opens.append(match['Open'].iloc[0])
        else:
            week_opens.append(df.loc[idx_dt, 'Open'])
    df['Week_Open'] = week_opens

    # Define Strategy Subpopulation Condition:
    # 1. Day of week is Monday (0) or Tuesday (1)
    # 2. Current price is below Weekly Open price (Discount Zone)
    strategy_mask = (df.index.dayofweek.isin([0, 1])) & (df['Price'] < df['Week_Open'])

    strategy_returns = df.loc[strategy_mask, 'Forward_7D_Return'].dropna()
    baseline_returns = df.loc[~strategy_mask, 'Forward_7D_Return'].dropna()

    print("======================================================================")
    print("      STATISTICAL HYPOTHESIS TESTING RESULTS (CryptoCast Theory)     ")
    print("======================================================================")
    print(f"Total Trading Days Analyzed: {len(df):,}")
    print(f"Strategy Entry Days (Mon/Tue Weekly Open Discount): {len(strategy_returns):,} samples")
    print(f"Baseline Days (All Other Market States)           : {len(baseline_returns):,} samples\n")

    # Descriptive Statistics
    mean_strat = strategy_returns.mean()
    std_strat = strategy_returns.std()
    mean_base = baseline_returns.mean()
    std_base = baseline_returns.std()

    print(f"Strategy Mean 7D Return  : +{mean_strat:.4f}% (Std: {std_strat:.2f}%)")
    print(f"Baseline Mean 7D Return  : +{mean_base:.4f}% (Std: {std_base:.2f}%)\n")

    # 1. Two-Sample Welch's t-Test (Unequal Variances)
    t_stat, p_val_ttest = stats.ttest_ind(strategy_returns, baseline_returns, equal_var=False, alternative='greater')
    
    # 2. Mann-Whitney U Test (Non-Parametric test robust to non-Gaussian financial tails)
    u_stat, p_val_mwu = stats.mannwhitneyu(strategy_returns, baseline_returns, alternative='greater')

    # 3. Effect Size (Cohen's d)
    pooled_std = np.sqrt(((len(strategy_returns)-1)*std_strat**2 + (len(baseline_returns)-1)*std_base**2) / (len(strategy_returns) + len(baseline_returns) - 2))
    cohens_d = (mean_strat - mean_base) / pooled_std

    print("----------------------------------------------------------------------")
    print("1. Two-Sample Welch's t-Test Results:")
    print(f"   t-Statistic        = {t_stat:.4f}")
    print(f"   p-Value (One-Tailed)= {p_val_ttest:.6f}")

    print("\n2. Mann-Whitney U Test Results (Non-Parametric):")
    print(f"   U-Statistic        = {u_stat:,.1f}")
    print(f"   p-Value (One-Tailed)= {p_val_mwu:.6f}")

    print(f"\n3. Effect Size (Cohen's d) = {cohens_d:.4f}")

    print("----------------------------------------------------------------------")
    print("CONCLUSION & DECISION:")
    if p_val_ttest < 0.05:
        print("  [ACCEPT H1] REJECT NULL HYPOTHESIS (H0) at alpha = 0.05 level!")
        print("  The Monday/Tuesday Weekly Open Discount strategy achieves STATISTICALLY SIGNIFICANT excess returns.")
    else:
        print("  [FAIL TO REJECT H0] Null Hypothesis (H0) holds at alpha = 0.05 level.")
        print("  Observed excess return difference is within random sampling noise.")

if __name__ == '__main__':
    run_hypothesis_testing()
