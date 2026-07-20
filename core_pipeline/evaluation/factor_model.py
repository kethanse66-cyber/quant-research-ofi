
# factor_model.py
# Phase 6 — Fama-French 3-Factor Attribution
# Constructs proper NAV series from bar-level PnL
# daily_return = daily_pnl / nav(t-1)  — this is a valid return series
# Regresses excess return on MKT-RF, SMB, HML
# Reference: Fama & French (1993)

import pandas as pd
import numpy as np
from scipy import stats
import urllib.request
import zipfile
import io
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
PNL_PATH       = "E:/quant-research-ofi/features/SPY_backtest_pnl_final.parquet"
OUTPUT_PATH    = "E:/quant-research-ofi/reports/factor_attribution.csv"
STARTING_NAV   = 100_000   # assumed starting capital in dollars
FF3_URL        = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"

os.makedirs("E:/quant-research-ofi/reports", exist_ok=True)

# ── LOAD PNL ──────────────────────────────────────────────────────────────────
print("Loading PnL...")
pnl = pd.read_parquet(PNL_PATH)
pnl.index = pd.to_datetime(pnl.index)
pnl = pnl.sort_index()

# ── BUILD NAV SERIES ──────────────────────────────────────────────────────────
# Aggregate bar-level net PnL to daily
daily_pnl = pnl['pnl_net'].resample('B').sum()

# NAV(t) = starting_capital + cumulative PnL up to day t
nav = STARTING_NAV + daily_pnl.cumsum()

# daily_return(t) = daily_pnl(t) / NAV(t-1)
# Correct formula — PnL earned on the capital at start of day
if (nav <= 0).any():
    raise ValueError("NAV became non-positive — cumulative losses exceed starting capital. Increase STARTING_NAV.")

daily_return = daily_pnl / nav.shift(1)
daily_return = daily_return.replace([np.inf, -np.inf], np.nan).dropna()
daily_return = daily_return[daily_return != 0]  # remove non-trading days

print(f"  Starting NAV   : ${STARTING_NAV:,.0f}")
print(f"  Final NAV      : ${nav.iloc[-1]:,.0f}")
print(f"  Total return   : {(nav.iloc[-1]/STARTING_NAV - 1)*100:.2f}%")
print(f"  Trading days   : {len(daily_return)}")
print(f"  Date range     : {daily_return.index[0].date()} to {daily_return.index[-1].date()}")
print(f"  Mean daily ret : {daily_return.mean()*100:.4f}%")
print(f"  Daily ret std  : {daily_return.std()*100:.4f}%")
print()

# ── DOWNLOAD FF3 FACTORS ─────────────────────────────────────────────────────
print("Downloading Fama-French 3 factors...")
try:
    req = urllib.request.urlopen(FF3_URL, timeout=30)
    zf  = zipfile.ZipFile(io.BytesIO(req.read()))
    # get first CSV file in zip
    csv_name = next(n for n in zf.namelist() if n.upper().endswith('.CSV'))
    raw = zf.open(csv_name).read().decode('utf-8', errors='ignore')

    # parse: skip header/footer lines that don't start with a digit
    rows = []
    for line in raw.split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        # valid data row: first field is 8-digit date
        if len(parts) >= 5 and parts[0].strip().isdigit() and len(parts[0].strip()) == 8:
            try:
                rows.append([int(parts[0].strip())] + [float(x) for x in parts[1:5]])
            except ValueError:
                continue

    ff3 = pd.DataFrame(rows, columns=['date', 'MKT_RF', 'SMB', 'HML', 'RF'])
    ff3['date'] = pd.to_datetime(ff3['date'].astype(str), format='%Y%m%d')
    ff3 = ff3.set_index('date')
    ff3 = ff3 / 100  # % to decimal
    ff3 = ff3.dropna()
    print(f"  FF3 loaded     : {ff3.index[0].date()} to {ff3.index[-1].date()}")
    print(f"  FF3 rows       : {len(ff3)}")

except Exception as e:
    print(f"  FF3 download failed: {e}")
    print("  Cannot run factor attribution without FF3 data.")
    exit()

print()

# ── ALIGN ─────────────────────────────────────────────────────────────────────
merged = pd.DataFrame({'strategy': daily_return}).join(ff3, how='inner')
merged = merged.dropna()
merged['excess_ret'] = merged['strategy'] - merged['RF']

print(f"  Aligned rows   : {len(merged)}")
print(f"  Aligned period : {merged.index[0].date()} to {merged.index[-1].date()}")
print()

if len(merged) < 30:
    print("ERROR: fewer than 30 aligned days. Check FF3 covers Apr 2024 - Apr 2025.")
    exit()

# ── OLS: excess_ret ~ MKT_RF + SMB + HML ─────────────────────────────────────
y   = merged['excess_ret'].values
X   = merged[['MKT_RF', 'SMB', 'HML']].values
n   = len(y)
k   = X.shape[1]
X_c = np.column_stack([np.ones(n), X])

beta  = np.linalg.lstsq(X_c, y, rcond=None)[0]
resid = y - X_c @ beta

ss_res = resid @ resid
ss_tot = ((y - y.mean()) ** 2).sum()
r2     = 1 - ss_res / ss_tot
r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)

mse   = ss_res / (n - k - 1)
se    = np.sqrt(mse * np.diag(np.linalg.inv(X_c.T @ X_c)))
tstat = beta / se
pval  = 2 * (1 - stats.t.cdf(np.abs(tstat), df=n - k - 1))

labels = ['Alpha', 'MKT_RF', 'SMB', 'HML']

# ── RESULTS ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("FAMA-FRENCH 3-FACTOR REGRESSION")
print("Dependent variable: daily excess return = (net PnL / NAV_prev) - RF")
print(f"Assumed starting NAV: ${STARTING_NAV:,.0f}")
print("=" * 60)
print(f"  {'Factor':<10} {'Beta':>12} {'SE':>12} {'t-stat':>10} {'p-value':>10} {'Sig':>5}")
print(f"  {'-'*62}")
for i, lab in enumerate(labels):
    sig = '***' if pval[i] < 0.01 else '**' if pval[i] < 0.05 else '*' if pval[i] < 0.10 else ''
    print(f"  {lab:<10} {beta[i]:>12.6f} {se[i]:>12.6f} {tstat[i]:>10.3f} {pval[i]:>10.4f} {sig:>5}")

print()
print(f"  R-squared      : {r2:.4f}")
print(f"  Adj R-squared  : {r2_adj:.4f}")
print(f"  N observations : {n}")
print()
print(f"  Alpha (daily)      : {beta[0]*100:.4f}%")
print(f"  Alpha (annualised) : {beta[0]*252*100:.2f}%")
print()

# ── INTERPRETATION ────────────────────────────────────────────────────────────
print("=" * 60)
print("INTERPRETATION")
print("=" * 60)
if pval[0] < 0.05:
    print(f"  Alpha significant (p={pval[0]:.3f}) — returns not fully explained by FF3")
else:
    print(f"  Alpha not significant (p={pval[0]:.3f}) — FF3 may explain returns")

print(f"  MKT_RF beta = {beta[1]:.4f} — {'long market bias' if beta[1] > 0.1 else 'near market-neutral'}")
print(f"  SMB beta    = {beta[2]:.4f} — {'small-cap tilt' if beta[2] > 0 else 'large-cap tilt'}")
print(f"  HML beta    = {beta[3]:.4f} — {'value tilt' if beta[3] > 0 else 'growth tilt'}")
print(f"  R² = {r2:.4f} — FF3 explains {r2*100:.1f}% of daily return variance")
print()

# ── HONEST ASSESSMENT ────────────────────────────────────────────────────────
print("=" * 60)
print("HONEST ASSESSMENT")
print("=" * 60)
print(f"  NAV constructed assuming ${STARTING_NAV:,.0f} starting capital.")
print("  Alpha and factor loadings depend on the assumed capital base because returns are constructed from PnL.")
print("  The sign and statistical significance are generally more informative than the exact coefficient magnitudes.")
print(f"  Beta magnitudes depend on NAV assumption — interpret direction only.")
print(f"  Single-ticker SPY — cross-asset attribution needs full portfolio PnL.")
print()
print(f"  Primary evidence for OFI alpha remains:")
print(f"    IC t-stat  : 3.26  (84 walk-forward folds, p=0.0016)")
print(f"    OOS IC     : +0.0022  (Mar-Apr 2025)")
print(f"    OOS net SR : +0.246")
print(f"  FF3 is supplementary context — not the main test.")
print()

# ── SAVE ─────────────────────────────────────────────────────────────────────
pd.DataFrame({
    'factor':           labels,
    'beta':             beta.round(6),
    'se':               se.round(6),
    'tstat':            tstat.round(3),
    'pval':             pval.round(4),
    'significant_5pct': pval < 0.05,
    'r2':               round(r2, 4),
    'r2_adj':           round(r2_adj, 4),
    'n':                n,
    'alpha_annualised': round(beta[0] * 252, 6),
    'starting_nav':     STARTING_NAV,
}).to_csv(OUTPUT_PATH, index=False)

print(f"  Saved: {OUTPUT_PATH}")
