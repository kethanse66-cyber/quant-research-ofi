import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading SPY data...")
df = pd.read_parquet('features/SPY_features.parquet')
df['ts'] = pd.to_datetime(df['ts'])
df = df.set_index('ts')

# ── 2. SELECT FEATURES TO TEST ────────────────────────────────────────────────
features = ['ofi', 'ofi_norm', 'queue_imbalance', 'spread', 
            'spread_change', 'realized_vol', 'trade_imbalance', 
            'kyle_lambda', 'amihud']

# ── 3. SAMPLE 5000 ROWS — normaltest needs reasonable sample ─────────────────
df_sample = df[features].dropna().sample(5000, random_state=42)

# ── 4. NORMALITY TEST ON RAW FEATURES ────────────────────────────────────────
print("\n" + "="*60)
print("NORMALITY TEST — RAW FEATURES")
print("="*60)
print(f"{'Feature':<20} {'Skewness':>10} {'Kurtosis':>10} {'P-value':>12} {'Result':>10}")
print("-"*60)

results = {}
for feat in features:
    data = df_sample[feat].values
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)
    stat, pval = stats.normaltest(data)
    passed = pval > 0.05
    results[feat] = {'skew': skew, 'kurt': kurt, 'pval': pval, 'passed': passed}
    result_str = "PASS" if passed else "FAIL"
    print(f"{feat:<20} {skew:>10.3f} {kurt:>10.3f} {pval:>12.6f} {result_str:>10}")

passed = sum(1 for r in results.values() if r['passed'])
failed = len(features) - passed
print(f"\nPASS: {passed} | FAIL: {failed}")

# ── 5. APPLY RANK TRANSFORM ───────────────────────────────────────────────────
print("\n" + "="*60)
print("APPLYING RANK TRANSFORM")
print("="*60)

df_ranked = df_sample.copy()
for feat in features:
    df_ranked[feat] = stats.rankdata(df_sample[feat]) / len(df_sample)

# ── 6. NORMALITY TEST AFTER RANK TRANSFORM ───────────────────────────────────
print("\nNORMALITY TEST — AFTER RANK TRANSFORM")
print("="*60)
print(f"{'Feature':<20} {'Skewness':>10} {'Kurtosis':>10} {'P-value':>12} {'Result':>10}")
print("-"*60)

for feat in features:
    data = df_ranked[feat].values
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)
    stat, pval = stats.normaltest(data)
    passed = pval > 0.05
    result_str = "PASS" if passed else "FAIL"
    print(f"{feat:<20} {skew:>10.3f} {kurt:>10.3f} {pval:>12.6f} {result_str:>10}")

# ── 7. PLOT BEFORE AND AFTER ──────────────────────────────────────────────────
print("\nPlotting before and after...")
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
plot_features = ['realized_vol', 'ofi_norm', 'spread']

for idx, feat in enumerate(plot_features):
    # Before
    axes[idx, 0].hist(df_sample[feat], bins=50, color='red', alpha=0.7)
    axes[idx, 0].set_title(f'{feat} — RAW')
    axes[idx, 0].set_ylabel('Frequency')
    
    # After
    axes[idx, 1].hist(df_ranked[feat], bins=50, color='green', alpha=0.7)
    axes[idx, 1].set_title(f'{feat} — RANK TRANSFORMED')

plt.tight_layout()
plt.savefig('reports/hmm_normality_test.png', dpi=150)
print("Plot saved to reports/hmm_normality_test.png")
plt.show()

print("\nhmm_normality_test.py COMPLETE")
