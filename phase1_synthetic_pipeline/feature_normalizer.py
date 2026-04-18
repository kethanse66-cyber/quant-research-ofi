# =============================================================
# feature_normalizer.py
# Phase 1 — Synthetic Pipeline
# Day 11: Three normalization methods for microstructure features
# =============================================================

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# SYNTHETIC ONLY — remove when using real Polygon data
np.random.seed(42)
n = 200

raw_ofi = pd.Series(
    np.random.standard_t(df=3, size=n) * 500,
    name='OFI'
)

df = pd.DataFrame({
    'ofi'            : raw_ofi,
    'queue_imbalance': np.random.uniform(-1, 1, n),
    'spread'         : np.abs(np.random.normal(0.02, 0.005, n)),
    'spread_change'  : np.random.normal(0, 0.003, n),
    'microprice'     : np.random.normal(450, 1, n),
    'trade_imbalance': np.random.uniform(-1, 1, n),
    'ofi_norm'       : np.random.normal(0, 1, n),
})
# SYNTHETIC ONLY — end


# -------------------------------------------------------------
# METHOD 1 — Z-Score
# Formula: z = (x - mean) / std
# Example: values=[10,50,30], mean=30, std=16.33
#          result=[-1.22, 1.22, 0.0]
# Unbounded — outliers stay as outliers.
# -------------------------------------------------------------

def zscore_normalize(series: pd.Series, window: int = None) -> pd.Series:
    """Z-score. window=None means full sample, int means rolling."""
    if window is None:
        return (series - series.mean()) / series.std()
    mu  = series.rolling(window, min_periods=1).mean()
    std = series.rolling(window, min_periods=1).std().replace(0, np.nan)
    return (series - mu) / std


# -------------------------------------------------------------
# METHOD 2 — Min-Max
# Formula: (x - min) / (max - min)
# Example: values=[10,50,30], min=10, max=50
#          result=[0.0, 1.0, 0.5]
# Bounded [0,1] but sensitive to outliers.
# -------------------------------------------------------------

def minmax_normalize(series: pd.Series, window: int = None) -> pd.Series:
    """Min-max scale to [0,1]. window=None full sample, int rolling."""
    if window is None:
        return (series - series.min()) / (series.max() - series.min())
    mn    = series.rolling(window, min_periods=1).min()
    mx    = series.rolling(window, min_periods=1).max()
    denom = (mx - mn).replace(0, np.nan)
    return (series - mn) / denom


# -------------------------------------------------------------
# METHOD 3 — Rank Transform  ← USE THIS BEFORE HMM
# Formula: rank(x) / N  → value between 0 and 1
# Example: values=[10,50,30]
#          ranks=[1,3,2] → result=[0.33, 1.0, 0.67]
#
# Why rank before HMM:
#   OFI and spread are skewed and fat-tailed.
#   Rank transform removes the shape entirely.
#   HMM only sees order (top 10%? bottom 50%?) not raw magnitude.
#   Z-score and min-max still preserve the original skew.
# -------------------------------------------------------------

def rank_transform(series: pd.Series, window: int = None) -> pd.Series:
    """Percentile rank in [0,1]. window=None full sample, int rolling."""
    if window is None:
        ranks = rankdata(series.values, method='average')
        return pd.Series(ranks / len(ranks), index=series.index)

    result = series.copy().astype(float)
    arr    = series.values
    for i in range(len(arr)):
        start       = max(0, i - window + 1)
        window_vals = arr[start : i + 1]
        r           = rankdata(window_vals, method='average')
        result.iloc[i] = r[-1] / len(window_vals)
    return result


# -------------------------------------------------------------
# NORMALIZE FULL DATAFRAME
# Applies chosen method to every column (or subset).
# -------------------------------------------------------------

def normalize_features(
    df       : pd.DataFrame,
    method   : str  = 'rank',
    window   : int  = None,
    columns  : list = None
) -> pd.DataFrame:
    """Apply zscore / minmax / rank to all columns or a subset."""
    method_map = {
        'zscore': zscore_normalize,
        'minmax': minmax_normalize,
        'rank'  : rank_transform,
    }
    if method not in method_map:
        raise ValueError(f"method must be one of {list(method_map.keys())}")

    fn   = method_map[method]
    cols = columns if columns is not None else df.columns.tolist()
    out  = df.copy()
    for col in cols:
        out[col] = fn(df[col], window=window)
    return out


# -------------------------------------------------------------
# AUDIT — before vs after stats for one feature
# Useful for paper Section 5 — shows skew and kurtosis change.
# -------------------------------------------------------------

def normalization_audit(original: pd.Series, normalized: pd.Series, label: str = "") -> None:
    """Print before/after stats for a single feature."""
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"  {'stat':<10} {'before':>12} {'after':>12}")
    print(f"  {'─'*36}")
    stats = {
        'mean': (original.mean(),  normalized.mean()),
        'std' : (original.std(),   normalized.std()),
        'min' : (original.min(),   normalized.min()),
        'max' : (original.max(),   normalized.max()),
        'skew': (original.skew(),  normalized.skew()),
        'kurt': (original.kurt(),  normalized.kurt()),
    }
    for name, (b, a) in stats.items():
        print(f"  {name:<10} {b:>12.4f} {a:>12.4f}")
    print(f"  NaN: before={original.isna().sum()}, after={normalized.isna().sum()}")


# =============================================================
# TESTS
# =============================================================

print("=" * 55)
print("TEST 1 — Rank: all values in [0, 1]")
r = rank_transform(df['ofi'])
assert r.min() >= 0.0 and r.max() <= 1.0, "FAIL"
print(f"  min={r.min():.4f}  max={r.max():.4f}  PASS")

print()
print("TEST 2 — Z-score: mean~0, std~1")
z = zscore_normalize(df['ofi'])
assert abs(z.mean()) < 1e-6, "FAIL mean"
assert abs(z.std(ddof=0) - 1.0) < 1e-6, "FAIL std"
print(f"  mean={z.mean():.6f}  std={z.std(ddof=0):.4f}  PASS")

print()
print("TEST 3 — Min-max: min=0, max=1")
m = minmax_normalize(df['ofi'])
assert abs(m.min()) < 1e-9 and abs(m.max() - 1.0) < 1e-9, "FAIL"
print(f"  min={m.min():.4f}  max={m.max():.4f}  PASS")

print()
print("TEST 4 — Rank on known array [10, 50, 30]")
test = pd.Series([10, 50, 30])
result   = rank_transform(test).round(4).tolist()
expected = [round(1/3, 4), round(3/3, 4), round(2/3, 4)]
print(f"  Expected: {expected}")
print(f"  Actual:   {result}")
assert result == expected, "FAIL"
print("  PASS")

print()
print("TEST 5 — Rolling rank window=50: no NaN after warmup")
rr = rank_transform(df['ofi'], window=50)
assert rr.isna().sum() == 0, "FAIL — unexpected NaN"
assert rr.min() >= 0.0 and rr.max() <= 1.0, "FAIL — out of range"
print(f"  NaN count=0  min={rr.min():.4f}  max={rr.max():.4f}  PASS")

# =============================================================
# COMPARISON TABLE — first 5 rows of OFI
# =============================================================

print()
print("=" * 55)
print("COMPARISON — first 5 rows, OFI column")
print(f"{'Raw OFI':>12} | {'Rank':>8} | {'Z-score':>8} | {'MinMax':>8}")
print("-" * 50)
for i in range(5):
    print(f"{df['ofi'].iloc[i]:>12.2f} | {r.iloc[i]:>8.4f} | {z.iloc[i]:>8.4f} | {m.iloc[i]:>8.4f}")

# =============================================================
# AUDIT — skew and kurtosis before vs after
# =============================================================

normalization_audit(df['ofi'], r, label="OFI → rank (full sample)")
normalization_audit(df['ofi'], z, label="OFI → z-score (full sample)")
normalization_audit(df['ofi'], rank_transform(df['ofi'], window=120),
                    label="OFI → rolling rank (window=120)")

# =============================================================
# NORMALIZE FULL FEATURE DATAFRAME
# =============================================================

df_rank_normalized = normalize_features(df, method='rank', window=120)
print("\n\nNormalized DataFrame — first 5 rows:")
print(df_rank_normalized.head())
print("\nAll values should be in [0, 1]:")
print(df_rank_normalized.describe().loc[['min', 'max']])
