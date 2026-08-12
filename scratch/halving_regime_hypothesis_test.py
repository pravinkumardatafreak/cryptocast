import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from scipy import stats
from src.macro_regime import classify_3state_macro_regimes

DATA_PATH = os.path.join('data', 'btc_data.csv')

def run_halving_regime_hypothesis_tests():
    """
    Executes Statistical Hypothesis Tests:
    1. Chi-Square Test of Independence (Chi2):
       H0: 3-State Macro Regimes (Bull, Consolidation, Bear) are INDEPENDENT of the 4-Year Halving Cycle Quarters.
       H1: 3-State Macro Regimes statistically ALIGN with 4-Year Halving Cycle Quarters.
    
    2. Welch's t-Test:
       H0: Returns in Post-Halving Quarter 1 (Progress 0-25%) == Returns in Bear Quarter 3 (Progress 50-75%).
       H1: Returns in Post-Halving Quarter 1 > Returns in Bear Quarter 3.
    """
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    # Classify 3-State Macro Regimes
    df = classify_3state_macro_regimes(df)

    # 4 Halving Cycle Quarters based on Halving_Progress
    def get_halving_quarter(progress):
        if progress < 0.25:
            return 'Halving Q1: Post-Halving Bull (0-25%)'
        elif progress < 0.50:
            return 'Halving Q2: Mid-Cycle Expansion (25-50%)'
        elif progress < 0.75:
            return 'Halving Q3: Late-Cycle Bear (50-75%)'
        else:
            return 'Halving Q4: Pre-Halving Accumulation (75-100%)'

    df['Halving_Quarter'] = df['Halving_Progress'].apply(get_halving_quarter)
    df['Daily_Return'] = df['Price'].pct_change() * 100.0
    df = df.dropna()

    print("======================================================================")
    print(" HYPOTHESIS TEST: 4-YEAR HALVING CYCLE ALIGNMENT WITH MACRO REGIMES")
    print("======================================================================\n")

    # Contingency Table: Halving Quarter vs 3-State Macro Regime
    contingency_table = pd.crosstab(df['Halving_Quarter'], df['Macro_Regime'])
    print("Contingency Table (Sample Counts):")
    print(contingency_table)
    print("\n----------------------------------------------------------------------")

    # 1. Chi-Square Test of Independence
    chi2, p_val_chi2, dof, expected = stats.chi2_contingency(contingency_table)

    print("1. Chi-Square Test of Independence Results:")
    print(f"   Chi2 Statistic     = {chi2:.4f}")
    print(f"   Degrees of Freedom = {dof}")
    print(f"   p-Value            = {p_val_chi2:.10e}")

    # 2. Welch's t-Test: Halving Q1 vs Halving Q3 Returns
    q1_returns = df.loc[df['Halving_Quarter'] == 'Halving Q1: Post-Halving Bull (0-25%)', 'Daily_Return']
    q3_returns = df.loc[df['Halving_Quarter'] == 'Halving Q3: Late-Cycle Bear (50-75%)', 'Daily_Return']

    t_stat, p_val_ttest = stats.ttest_ind(q1_returns, q3_returns, equal_var=False, alternative='greater')

    print("\n2. Two-Sample Welch's t-Test Results (Halving Q1 vs Halving Q3 Returns):")
    print(f"   Halving Q1 Mean Daily Return = +{q1_returns.mean():.4f}% (n = {len(q1_returns):,})")
    print(f"   Halving Q3 Mean Daily Return = +{q3_returns.mean():.4f}% (n = {len(q3_returns):,})")
    print(f"   t-Statistic                  = {t_stat:.4f}")
    print(f"   p-Value (One-Tailed)         = {p_val_ttest:.8f}")

    print("\n----------------------------------------------------------------------")
    print("STATISTICAL DECISION & INTERPRETATION:")
    if p_val_chi2 < 0.05:
        print("  [REJECT H0 IN CHI2 TEST]: p-value < 0.05")
        print("  The 4-Year Halving Cycle Quarters statistically ALIGN with 3-State Macro Regimes.")
    
    if p_val_ttest < 0.05:
        print("  [REJECT H0 IN T-TEST]: p-value < 0.05")
        print("  Post-Halving Q1 daily returns are STATISTICALLY SIGNIFICANTLY HIGHER than Late-Cycle Q3 returns.")

if __name__ == '__main__':
    run_halving_regime_hypothesis_tests()
