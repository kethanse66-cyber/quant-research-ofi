import pandas as pd
import numpy as np

df = pd.read_parquet("E:/quant-research-ofi/features/walk_forward_results.parquet")

# Last 4 folds only — post burn-in period
last_4 = df[df['test_month'].isin(['2025-01', '2025-02', '2025-03', '2025-04'])]

print("POST BURN-IN ANALYSIS — Last 4 folds only")
print("=" * 50)

summary = last_4.groupby('ticker').agg(
    gross_sharpe  = ('sharpe_gross', 'mean'),
    net_sharpe    = ('sharpe_net',   'mean'),
    mean_ic       = ('ic',           'mean'),
    survival_rate = ('survives',     'mean'),
    cost_drag     = ('cost_pct',     'mean')
).round(3).sort_values('net_sharpe', ascending=False)

summary['survival_pct'] = (summary['survival_rate'] * 100).round(1)
print(summary[['gross_sharpe', 'net_sharpe', 'mean_ic', 'survival_pct', 'cost_drag']].to_string())

print()
print(f"Mean gross Sharpe (last 4 folds) : {last_4['sharpe_gross'].mean():.3f}")
print(f"Mean net   Sharpe (last 4 folds) : {last_4['sharpe_net'].mean():.3f}")
print(f"Survival rate     (last 4 folds) : {last_4['survives'].mean()*100:.1f}%")
print(f"Mean IC           (last 4 folds) : {last_4['ic'].mean():.4f}")
