import pandas as pd
import numpy as np

df = pd.read_parquet("E:/quant-research-ofi/features/walk_forward_results.parquet")

print("REGIME ANALYSIS — does signal improve over time?")
print("(later folds = more training data = should improve)")
print()
pivot = df.pivot_table(
    index='test_month',
    columns='ticker', 
    values='sharpe_gross'
).sort_index()

print("Mean gross Sharpe across all tickers per fold:")
fold_mean = pivot.mean(axis=1)
for month, val in fold_mean.items():
    bar = "█" * int(abs(val)) if val > 0 else "▒" * int(abs(val))
    print(f"  {month}  {val:+.3f}  {bar}")

print()
print(f"First 3 folds avg : {fold_mean.iloc[:3].mean():.3f}")
print(f"Last  4 folds avg : {fold_mean.iloc[3:].mean():.3f}")
print()
print("Correlation of fold number with mean Sharpe:")
corr = np.corrcoef(range(len(fold_mean)), fold_mean.values)[0,1]
print(f"  corr = {corr:.3f}")
if corr > 0.3:
    print("  Positive trend — signal improves with more training data")
elif corr < -0.3:
    print("  Negative trend — signal degrades over time")
else:
    print("  No clear trend — signal stable but noisy")
