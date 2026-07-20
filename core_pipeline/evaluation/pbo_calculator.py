# pbo_calculator.py
# Phase 6 — PBO-style Robustness Test
# Method: Combinatorially Symmetric Cross-Validation (CSCV)
# Reference: Bailey & Lopez de Prado (2014)
#
# NOTE ON METHODOLOGY:
# True PBO requires multiple strategy specifications on the same asset.
# We have one model (Ridge) tested across 12 tickers x 7 folds.
# We therefore implement a cross-sectional CSCV robustness test:
#   "Does the best-performing ticker in training generalise to test?"
# This is PBO-inspired but not identical to Bailey & Lopez de Prado.
# Reported as "PBO-style robustness analysis" in paper Section 8.

import pandas as pd
import numpy as np
from scipy.stats import rankdata
from itertools import combinations
import os
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
PRED_PATH   = "E:/quant-research-ofi/features/walk_forward_results.parquet"
OUTPUT_PATH = "E:/quant-research-ofi/reports/pbo_results.csv"

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading walk-forward results...")
df = pd.read_parquet(PRED_PATH)
print(f"Shape   : {df.shape}")
print(f"Columns : {df.columns.tolist()}")
print(f"Tickers : {df['ticker'].unique().tolist()}")
print()

# ── BUILD PERFORMANCE MATRIX ──────────────────────────────────────────────────
# Rows = folds (test_month) — time dimension
# Columns = tickers — cross-sectional dimension
# Values = sharpe_gross — performance metric
#
# Limitation acknowledged: tickers are assets, not strategy specifications.
# This tests cross-sectional generalisation, not model overfitting per se.
# See paper Section 8 for full qualification.

print("=" * 60)
print("PERFORMANCE MATRIX")
print("=" * 60)

perf_matrix = df.pivot_table(
    index='test_month',
    columns='ticker',
    values='sharpe_gross'
).sort_index().dropna()

T, N = perf_matrix.shape
print(f"  Shape : {T} folds x {N} tickers")
print(f"  Folds : {perf_matrix.index.tolist()}")
print()
print(perf_matrix.round(3).to_string())
print()

if T < 4:
    raise ValueError(f"Only {T} folds — need at least 4 for CSCV.")

# ── CSCV ──────────────────────────────────────────────────────────────────────
# Steps:
# 1. Generate all combinations of ceil(T/2) folds for training
#    Remaining folds = test set
# 2. For each split:
#    a. Find best ticker in train: n* = argmax(mean Sharpe)
#    b. Rank n* in test set by mean Sharpe
#    c. Logit of relative rank: w = log(rank / (N+1-rank))
# 3. PBO = fraction of splits where w < 0
#    (best train ticker ranked below median in test)
#
# Problem 2 fix: S is removed entirely.
# We use ceil(T/2) directly — all combinations of half the folds.
# This is the correct CSCV implementation for our fold count.

half_t     = T // 2
all_combos = list(combinations(range(T), half_t))

print("=" * 60)
print(f"CSCV — {len(all_combos)} splits (C({T},{half_t}))")
print("=" * 60)
print(f"  Train folds per split : {half_t}")
print(f"  Test  folds per split : {T - half_t}")
print(f"  Total splits          : {len(all_combos)}")
print()

logit_values     = []
overfit_count    = 0
selected_oos     = []
average_oos      = []
selection_counts = {col: 0 for col in perf_matrix.columns}

for combo in all_combos:
    train_idx = list(combo)
    test_idx  = [i for i in range(T) if i not in train_idx]

    train_perf = perf_matrix.iloc[train_idx]
    test_perf  = perf_matrix.iloc[test_idx]

    # a. Best ticker in train
    train_sharpe  = train_perf.mean(axis=0)
    best_ticker   = train_sharpe.idxmax()
    selection_counts[best_ticker] += 1

    # b. Rank best ticker in test
    test_sharpe = test_perf.mean(axis=0)
    test_ranked = rankdata(test_sharpe)   # rank 1=worst, N=best
    best_pos    = perf_matrix.columns.get_loc(best_ticker)
    best_rank   = test_ranked[best_pos]

    # c. Logit of relative rank
    # Use N+1 in denominator to avoid log(0) at boundaries
    relative_rank = best_rank / (N + 1)
    relative_rank = np.clip(relative_rank, 1e-6, 1 - 1e-6)
    w = np.log(relative_rank / (1 - relative_rank))

    logit_values.append(w)
    if w < 0:
        overfit_count += 1

    # OOS performance tracking
    selected_oos.append(test_sharpe[best_ticker])
    average_oos.append(test_sharpe.mean())

# ── PBO SCORE ─────────────────────────────────────────────────────────────────
n_splits = len(logit_values)
pbo      = overfit_count / n_splits if n_splits > 0 else np.nan
logit_arr = np.array(logit_values)

print("=" * 60)
print("PBO-STYLE ROBUSTNESS SCORE")
print("=" * 60)
print(f"  Splits evaluated : {n_splits}")
print(f"  Overfit count    : {overfit_count}")
print(f"  PBO score        : {pbo:.4f}")
print()

# Problem 3 acknowledged in interpretation
if pbo <= 0.10:
    label = "Very low overfitting — strong cross-sectional generalisation"
elif pbo <= 0.25:
    label = "Low overfitting — results likely genuine"
elif pbo <= 0.40:
    label = "Moderate overfitting — interpret with caution"
elif pbo <= 0.55:
    label = "High overfitting — near random selection"
else:
    label = "Severe overfitting — selection unreliable"

print(f"  Interpretation   : {label}")
print()
print(f"  CAVEAT: T={T} folds produces only {n_splits} splits.")
print(f"  Estimate is noisy. Bailey & Lopez de Prado recommend T >> 16.")
print(f"  Report as PBO-style robustness analysis, not true PBO.")
print()

# ── LOGIT DISTRIBUTION ────────────────────────────────────────────────────────
print("=" * 60)
print("LOGIT DISTRIBUTION")
print("=" * 60)
print(f"  Mean : {logit_arr.mean():.4f}")
print(f"  Std  : {logit_arr.std():.4f}")
print(f"  Min  : {logit_arr.min():.4f}")
print(f"  Max  : {logit_arr.max():.4f}")
print(f"  Pct negative (= PBO) : {(logit_arr < 0).mean()*100:.1f}%")
print()

# ── SELECTION FREQUENCY ───────────────────────────────────────────────────────
print("=" * 60)
print("TICKER SELECTION FREQUENCY IN TRAINING")
print("=" * 60)
total_sel = sum(selection_counts.values())
for ticker, count in sorted(selection_counts.items(), key=lambda x: -x[1]):
    pct = count / total_sel * 100
    print(f"  {ticker:<8} selected {count:>4}x ({pct:.1f}%)")
print()

# ── OOS PERFORMANCE ───────────────────────────────────────────────────────────
mean_selected = np.mean(selected_oos)
mean_average  = np.mean(average_oos)
excess        = mean_selected - mean_average

print("=" * 60)
print("OOS: SELECTED TICKER vs AVERAGE TICKER")
print("=" * 60)
print(f"  Mean OOS Sharpe (selected) : {mean_selected:.4f}")
print(f"  Mean OOS Sharpe (average)  : {mean_average:.4f}")
print(f"  Excess                     : {excess:.4f}")
if excess > 0:
    print(f"  Positive → selection adds value OOS")
else:
    print(f"  Negative → selected ticker underperforms average OOS")
print()

# ── SAVE ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SUMMARY")
print("=" * 60)
results = {
    "n_folds":           T,
    "n_tickers":         N,
    "n_splits":          n_splits,
    "half_t":            half_t,
    "overfit_count":     overfit_count,
    "pbo":               round(pbo, 4),
    "pbo_label":         label,
    "mean_logit":        round(logit_arr.mean(), 4),
    "std_logit":         round(logit_arr.std(), 4),
    "mean_oos_selected": round(mean_selected, 4),
    "mean_oos_average":  round(mean_average, 4),
    "excess_oos":        round(excess, 4),
    "caveat":            f"T={T} folds only — noisy estimate. PBO-style not true PBO."
}
for k, v in results.items():
    print(f"  {k:<25} : {v}")

os.makedirs("E:/quant-research-ofi/reports", exist_ok=True)
pd.DataFrame([results]).to_csv(OUTPUT_PATH, index=False)
print(f"\n  Saved to: {OUTPUT_PATH}")
