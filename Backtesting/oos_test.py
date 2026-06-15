# oos_test.py
# Phase 6 — True Out-of-Sample Test
# Method: Time-based OOS — last 2 months (Mar-Apr 2025) never seen during development
# All 12 tickers evaluated on same held-out period
# Reference: Lopez de Prado (2018) Ch.7

import pandas as pd
import numpy as np
from scipy import stats
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
PRED_PATH   = "E:/quant-research-ofi/features/walk_forward_results.parquet"
OUTPUT_PATH = "E:/quant-research-ofi/reports/oos_results.csv"

# True OOS period — last 2 folds only
# IMPORTANT: These months are valid OOS only if Ridge alpha was fixed before
# any walk-forward runs and no parameters were tuned after seeing these months.
# If hyperparameters were adjusted after viewing Mar-Apr results, this becomes
# a robustness check, not a true OOS test.
OOS_MONTHS  = ['2025-03', '2025-04']
IS_MONTHS   = ['2024-10', '2024-11', '2024-12', '2025-01', '2025-02']

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading walk-forward results...")
df = pd.read_parquet(PRED_PATH)
print(f"Shape   : {df.shape}")
print(f"Tickers : {df['ticker'].unique().tolist()}")
print()

# ── OOS WARNING ───────────────────────────────────────────────────────────────
print("=" * 60)
print("WARNING")
print("=" * 60)
print(f"  OOS period contains only {len(OOS_MONTHS)} months ({OOS_MONTHS}).")
print(f"  Results should be interpreted cautiously.")
print(f"  A robust OOS test requires at least 6 months.")
print()

# ── SPLIT IS vs OOS ───────────────────────────────────────────────────────────
is_df  = df[df['test_month'].isin(IS_MONTHS)].copy()
oos_df = df[df['test_month'].isin(OOS_MONTHS)].copy()

print("=" * 60)
print("DATA SPLIT")
print("=" * 60)
print(f"  In-sample  months : {IS_MONTHS}")
print(f"  OOS        months : {OOS_MONTHS}")
print(f"  IS  rows   : {len(is_df)}  ({len(IS_MONTHS)} months x {df['ticker'].nunique()} tickers)")
print(f"  OOS rows   : {len(oos_df)} ({len(OOS_MONTHS)} months x {df['ticker'].nunique()} tickers)")
print()

# ── IS vs OOS COMPARISON — ALL TICKERS ───────────────────────────────────────
print("=" * 60)
print("IS vs OOS — ALL METRICS PER TICKER")
print("=" * 60)

rows = []

for ticker in sorted(df['ticker'].unique()):
    is_t  = is_df[is_df['ticker']  == ticker]
    oos_t = oos_df[oos_df['ticker'] == ticker]

    is_ic    = is_t['ic'].mean()
    oos_ic   = oos_t['ic'].mean()
    is_sh    = is_t['sharpe_gross'].mean()
    oos_sh   = oos_t['sharpe_gross'].mean()
    is_net   = is_t['sharpe_net'].mean()
    oos_net  = oos_t['sharpe_net'].mean()
    is_surv  = is_t['survives'].mean()
    oos_surv = oos_t['survives'].mean()

    ic_decay   = oos_ic - is_ic
    sh_decay   = oos_sh - is_sh

    # Threshold: IC must be meaningfully positive AND Sharpe must be economically
    # significant (>0.5). Avoids counting near-zero results as "generalises".
    generalises = (
        oos_ic > 0
        and oos_sh > 0.5
    )

    rows.append({
        'ticker':      ticker,
        'is_ic':       round(is_ic,   4),
        'oos_ic':      round(oos_ic,  4),
        'ic_decay':    round(ic_decay, 4),
        'is_sharpe':   round(is_sh,   3),
        'oos_sharpe':  round(oos_sh,  3),
        'sh_decay':    round(sh_decay, 3),
        'is_net':      round(is_net,  3),
        'oos_net':     round(oos_net, 3),
        'is_surv':     round(is_surv,  2),
        'oos_surv':    round(oos_surv, 2),
        'generalises': generalises,
    })

result_df = pd.DataFrame(rows).sort_values('oos_sharpe', ascending=False)
print(result_df.to_string(index=False))
print()

# ── AGGREGATE IS vs OOS ───────────────────────────────────────────────────────
print("=" * 60)
print("AGGREGATE IS vs OOS")
print("=" * 60)

agg_is_ic    = is_df['ic'].mean()
agg_oos_ic   = oos_df['ic'].mean()
agg_is_sh    = is_df['sharpe_gross'].mean()
agg_oos_sh   = oos_df['sharpe_gross'].mean()
agg_is_net   = is_df['sharpe_net'].mean()
agg_oos_net  = oos_df['sharpe_net'].mean()
agg_is_surv  = is_df['survives'].mean()
agg_oos_surv = oos_df['survives'].mean()

print(f"  {'Metric':<25} {'In-Sample':>12} {'OOS':>12} {'Decay':>12}")
print(f"  {'-'*60}")
print(f"  {'Mean IC':<25} {agg_is_ic:>12.4f} {agg_oos_ic:>12.4f} {agg_oos_ic-agg_is_ic:>12.4f}")
print(f"  {'Mean Gross Sharpe':<25} {agg_is_sh:>12.3f} {agg_oos_sh:>12.3f} {agg_oos_sh-agg_is_sh:>12.3f}")
print(f"  {'Mean Net Sharpe':<25} {agg_is_net:>12.3f} {agg_oos_net:>12.3f} {agg_oos_net-agg_is_net:>12.3f}")
print(f"  {'Survival Rate':<25} {agg_is_surv:>12.1%} {agg_oos_surv:>12.1%} {agg_oos_surv-agg_is_surv:>12.1%}")
print()

# ── STATISTICAL TEST: IS IC vs OOS IC ────────────────────────────────────────
# H0: mean(OOS IC) == mean(IS IC)
# Welch t-test — correct choice when IS and OOS sample sizes differ

print("=" * 60)
print("STATISTICAL TEST: IS IC vs OOS IC (Welch t-test)")
print("=" * 60)
tstat, pval = stats.ttest_ind(
    oos_df['ic'].dropna(),
    is_df['ic'].dropna(),
    equal_var=False
)
print(f"  IS  mean IC  : {agg_is_ic:.4f}  (n={len(is_df)})")
print(f"  OOS mean IC  : {agg_oos_ic:.4f}  (n={len(oos_df)})")
print(f"  t-stat       : {tstat:.3f}")
print(f"  p-value      : {pval:.4f}")
if pval > 0.05:
    print(f"  Result       : No significant difference IS vs OOS (p={pval:.3f})")
    print(f"  Interpretation: IC is stable — no evidence of overfitting to IS period")
else:
    print(f"  Result       : Significant difference IS vs OOS (p={pval:.3f})")
    print(f"  Interpretation: IC changed significantly — investigate overfitting")
print()

# ── GENERALISATION SCORE ─────────────────────────────────────────────────────
print("=" * 60)
print("GENERALISATION SCORE  (threshold: OOS IC > 0 AND OOS Sharpe > 0.5)")
print("=" * 60)
n_generalise = result_df['generalises'].sum()
n_total      = len(result_df)
gen_pct      = n_generalise / n_total * 100

print(f"  Tickers with OOS IC > 0 AND OOS Sharpe > 0.5:")
for _, row in result_df[result_df['generalises']].iterrows():
    print(f"    {row['ticker']:<8} OOS IC={row['oos_ic']:+.4f}  OOS Sharpe={row['oos_sharpe']:+.3f}")
print()
print(f"  Generalisation rate: {n_generalise}/{n_total} tickers ({gen_pct:.1f}%)")
print()

# ── OOS MONTH BY MONTH ────────────────────────────────────────────────────────
print("=" * 60)
print("OOS MONTH BY MONTH")
print("=" * 60)
for month in OOS_MONTHS:
    month_df = oos_df[oos_df['test_month'] == month]
    print(f"  {month} | Mean IC={month_df['ic'].mean():+.4f} | "
          f"Mean Gross SR={month_df['sharpe_gross'].mean():+.3f} | "
          f"Mean Net SR={month_df['sharpe_net'].mean():+.3f} | "
          f"Survival={month_df['survives'].mean():.0%}")
print()

# ── HONEST ASSESSMENT ────────────────────────────────────────────────────────
print("=" * 60)
print("HONEST ASSESSMENT")
print("=" * 60)
if agg_oos_ic > 0:
    print(f"  OOS IC positive ({agg_oos_ic:.4f}) — signal generalises directionally")
else:
    print(f"  OOS IC negative ({agg_oos_ic:.4f}) — signal does not generalise")

if agg_oos_sh > agg_is_sh:
    print(f"  OOS Sharpe HIGHER than IS ({agg_oos_sh:.3f} vs {agg_is_sh:.3f}) — improving trend")
elif agg_oos_sh > 0:
    print(f"  OOS Sharpe positive but lower than IS — moderate decay")
else:
    print(f"  OOS Sharpe negative ({agg_oos_sh:.3f}) — signal does not survive OOS gross")

if agg_oos_net > 0:
    print(f"  OOS Net Sharpe positive ({agg_oos_net:.3f}) — survives costs OOS")
else:
    print(f"  OOS Net Sharpe negative ({agg_oos_net:.3f}) — costs kill signal OOS")

print(f"  {n_generalise}/{n_total} tickers generalise (OOS IC > 0 AND OOS Sharpe > 0.5)")
print(f"  Note: OOS window is only 2 months — interpret with caution")
print()

# ── SAVE ─────────────────────────────────────────────────────────────────────
os.makedirs("E:/quant-research-ofi/reports", exist_ok=True)
result_df.to_csv(OUTPUT_PATH, index=False)

summary_row = pd.DataFrame([{
    'is_ic':        round(agg_is_ic,   4),
    'oos_ic':       round(agg_oos_ic,  4),
    'ic_decay':     round(agg_oos_ic - agg_is_ic, 4),
    'is_sharpe':    round(agg_is_sh,   3),
    'oos_sharpe':   round(agg_oos_sh,  3),
    'is_net':       round(agg_is_net,  3),
    'oos_net':      round(agg_oos_net, 3),
    'is_survival':  round(agg_is_surv, 3),
    'oos_survival': round(agg_oos_surv,3),
    'generalisation_rate': round(gen_pct, 1),
    'tstat':        round(tstat, 3),
    'pval':         round(pval,  4),
    'oos_months':   str(OOS_MONTHS),
}])
summary_row.to_csv(
    OUTPUT_PATH.replace('oos_results', 'oos_summary'),
    index=False
)

print(f"  Saved: {OUTPUT_PATH}")
print(f"  Saved: {OUTPUT_PATH.replace('oos_results', 'oos_summary')}")
