import pandas as pd
import numpy as np
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
TICKER = "SPY"
FEATURE_PATH = f"E:/quant-research-ofi/features/{TICKER}_features.parquet"
REGIME_PATH  = f"E:/quant-research-ofi/features/{TICKER}_rolling_regimes.parquet"
OUTPUT_PATH  = "E:/quant-research-ofi/reports/audit_report.txt"

FEATURES = [
    "ofi", "ofi_10s", "ofi_30s", "ofi_1m", "ofi_5m", "ofi_10m",
    "ofi_norm", "queue_imbalance", "trade_imbalance",
    "spread", "spread_change", "microprice", "vwap",
    "kyle_lambda", "amihud", "realized_vol"
]

TARGET_COLS = ["fwd_ret_1m", "fwd_ret_5m", "fwd_ret_10m"]

# Bars per horizon — 10s bars
# 1 min  = 60 / 10 = 6 bars
# 5 min  = 300 / 10 = 30 bars
# 10 min = 600 / 10 = 60 bars
HORIZON_BARS = {
    "fwd_ret_1m":  6,
    "fwd_ret_5m":  30,
    "fwd_ret_10m": 60,   # Problem 2 fix — was wrongly 6 before
}

# ── LOAD DATA ────────────────────────────────────────────────────────────────
print("Loading feature file...")
df = pd.read_parquet(FEATURE_PATH)
df = df.sort_index()
print(f"Rows loaded: {len(df)}")
print(f"Date range: {df.index[0]} → {df.index[-1]}")
print()

# ── CHECK 1: TIMESTAMP INTEGRITY ─────────────────────────────────────────────
print("=" * 60)
print("CHECK 1 — TIMESTAMP INTEGRITY")
print("=" * 60)
is_monotonic = df.index.is_monotonic_increasing
n_dupes      = df.index.duplicated().sum()
print(f"  Monotonic increasing : {'OK' if is_monotonic else 'WARNING — index not sorted!'}")
print(f"  Duplicate timestamps : {'OK — zero duplicates' if n_dupes == 0 else 'WARNING — ' + str(n_dupes) + ' duplicates!'}")
print()

# ── CHECK 2: SHIFT(1) PRE-APPLIED? ───────────────────────────────────────────
# Problem 1 fix — removed autocorrelation test entirely.
# High autocorr is natural for microprice, vwap, realized_vol — not a reliable signal.
# Instead: only check that first row is not NaN AND spot-check one raw value makes sense.
# The real lookahead protection is the leakage correlation check in Check 4.

print("=" * 60)
print("CHECK 2 — FORWARD RETURNS IN FEATURE FILE? (should be NO)")
print("=" * 60)
for col in TARGET_COLS:
    if col in df.columns:
        print(f"  {col:<20} → WARNING: forward return found in feature file!")
    else:
        print(f"  {col:<20} → OK — not in feature file")
print()

# ── CHECK 3: ROLLING FEATURES BURN-IN ────────────────────────────────────────
print("=" * 60)
print("CHECK 3 — ROLLING FEATURES: LEADING NaN COUNT (burn-in)")
print("=" * 60)
rolling_feats = ["ofi_30s", "ofi_1m", "ofi_5m", "ofi_10m", "ofi_norm",
                 "kyle_lambda", "amihud", "realized_vol"]

for feat in rolling_feats:
    if feat not in df.columns:
        print(f"  {feat:<20} → MISSING")
        continue
    series  = df[feat]
    all_nan  = series.isna().all()
    none_nan = series.isna().sum() == 0

    if all_nan:
        print(f"  {feat:<20} → WARNING: ALL values are NaN!")
        continue
    if none_nan:
        print(f"  {feat:<20} → WARNING: zero NaN — no burn-in detected. Check window logic.")
        continue

    first_valid_pos = series.first_valid_index()
    n_leading       = df.index.get_loc(first_valid_pos)
    print(f"  {feat:<20} → OK — {n_leading} leading NaN rows (burn-in)")
print()

# ── CHECK 4: LEAKAGE CORRELATION — ALL HORIZONS ───────────────────────────────
# Problem 2 fix — shift(-60) for 10min, not shift(-6).
# Problem 3 add — sort by abs_corr and print top 5 worst offenders per horizon.
# If corr(feature[t], fwd_ret[t]) is high → feature contains future info → leakage.
# Flag threshold: |corr| > 0.05 (conservative for microstructure features).

print("=" * 60)
print("CHECK 4 — LEAKAGE CORRELATION: feature vs future return (all horizons)")
print("=" * 60)
print("  Price proxy: microprice")
print("  Flag threshold: |corr| > 0.05")
print()

if "microprice" not in df.columns:
    print("  microprice not found — skipping leakage check")
else:
    for horizon_name, n_bars in HORIZON_BARS.items():
        print(f"  ── Horizon: {horizon_name} (shift = -{n_bars} bars) ──")

        fwd_ret = np.log(df["microprice"].shift(-n_bars) / df["microprice"])

        rows = []
        for feat in FEATURES:
            if feat not in df.columns:
                continue
            valid = df[feat].notna() & fwd_ret.notna()
            if valid.sum() < 100:
                continue
            corr     = df.loc[valid, feat].corr(fwd_ret[valid])
            abs_corr = abs(corr)
            flag     = "WARNING — possible leakage!" if abs_corr > 0.05 else "OK"
            rows.append({
                "feature":  feat,
                "corr":     corr,
                "abs_corr": abs_corr,
                "flag":     flag
            })

        leakage_table = pd.DataFrame(rows).sort_values("abs_corr", ascending=False)

        # Print all features
        for _, row in leakage_table.iterrows():
            print(f"    {row['feature']:<20} corr={row['corr']:+.4f}  [{row['flag']}]")

        # Print top 5 worst offenders
        print()
        print(f"  Top 5 highest |corr| for {horizon_name}:")
        for _, row in leakage_table.head(5).iterrows():
            print(f"    {row['feature']:<20} |corr|={row['abs_corr']:.4f}")
        print()

# ── CHECK 5: FIRST VALID PREDICTION TIMESTAMP ────────────────────────────────
print("=" * 60)
print("CHECK 5 — FIRST VALID PREDICTION TIMESTAMP")
print("=" * 60)
available_feats = [f for f in FEATURES if f in df.columns]
first_valid_idx = df[available_feats].dropna().index[0]
n_burnin        = df.index.get_loc(first_valid_idx)
print(f"  First valid prediction timestamp : {first_valid_idx}")
print(f"  Rows burned in (unusable)        : {n_burnin}")
print(f"  Usable rows for training/testing : {len(df) - n_burnin}")
print()

# ── CHECK 6: REGIME LOOKAHEAD ────────────────────────────────────────────────
print("=" * 60)
print("CHECK 6 — REGIME FILE LOOKAHEAD CHECK")
print("=" * 60)
try:
    reg = pd.read_parquet(REGIME_PATH)
    reg = reg.sort_index()
    pct_matched = reg.index.isin(df.index).mean() * 100
    print(f"  Regime timestamps in feature index : {pct_matched:.1f}%")

    if reg.index.max() <= df.index.max():
        print(f"  Regime max ts ({reg.index.max()}) ≤ feature max → OK")
    else:
        print(f"  WARNING: regime has timestamps beyond feature file!")

    burn_in_end  = df.index[0] + pd.Timedelta(days=60)
    regime_start = reg.first_valid_index() if reg.isnull().any().any() else reg.index[0]
    print(f"  Regime first label : {regime_start}")
    print(f"  60-day burn-in end : {burn_in_end}")
    if regime_start >= burn_in_end:
        print(f"  Burn-in respected → OK")
    else:
        print(f"  WARNING: regime labels start before 60-day burn-in!")

except FileNotFoundError:
    print(f"  Regime file not found at: {REGIME_PATH} — skipping.")
print()

# ── CHECK 7: NaN SUMMARY ─────────────────────────────────────────────────────
print("=" * 60)
print("CHECK 7 — NaN SUMMARY PER FEATURE")
print("=" * 60)
for feat in FEATURES:
    if feat not in df.columns:
        print(f"  {feat:<20} → MISSING from file")
        continue
    n_nan   = df[feat].isna().sum()
    pct_nan = n_nan / len(df) * 100
    flag    = "WARNING" if pct_nan > 1.0 else "OK"
    print(f"  {feat:<20} → {n_nan:>6} NaN ({pct_nan:.2f}%) [{flag}]")
print()

# ── FINAL VERDICT ─────────────────────────────────────────────────────────────
print("=" * 60)
print("FINAL VERDICT")
print("=" * 60)
print(f"  Features available        : {len(available_feats)} / {len(FEATURES)}")
print(f"  First valid prediction ts : {first_valid_idx}")
print(f"  Usable rows               : {len(df) - n_burnin}")
print()
print("  All checks OK → audit PASSED. Safe to run backtest_engine.py")
print("  Any WARNING → investigate before backtesting.")

os.makedirs("E:/quant-research-ofi/reports", exist_ok=True)
print(f"\n  Save this output to: {OUTPUT_PATH}")
