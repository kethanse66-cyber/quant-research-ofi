# dsr_final.py
# Deflated Sharpe Ratio for the walk-forward strategy (paper Section 7.1).
# Input: features/wf_daily_pnl.parquet, produced by wf_daily_dsr.py.
import numpy as np
import pandas as pd
from scipy import stats

daily = pd.read_parquet(r"E:/quant-research-ofi/features/wf_daily_pnl.parquet")
idx = pd.Index([pd.Timestamp(t).tz_localize(None) if pd.Timestamp(t).tzinfo is None
                else pd.Timestamp(t).tz_convert("America/New_York").tz_localize(None)
                for t in daily.index]).normalize()
daily.index = idx
dup = daily.reset_index().duplicated(subset=["index", "ticker"]).sum()
print(f"rows={len(daily)}  tickers={daily['ticker'].nunique()}  duplicate (date,ticker)={dup}")

def dsr(r, n_trials):
    r = np.asarray(r, float); n = len(r)
    sr = r.mean() / r.std(ddof=1)
    sk = stats.skew(r); ku = stats.kurtosis(r, fisher=False)
    e = 0.5772156649
    sr0 = 0.0 if n_trials == 1 else np.sqrt(1 / n) * ((1 - e) * stats.norm.ppf(1 - 1 / n_trials)
                                                    + e * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    z = (sr - sr0) * np.sqrt(n - 1) / np.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr ** 2, 1e-12))
    return n, sr * np.sqrt(252), sk, ku - 3, stats.norm.cdf(z)

# Trial counts: 1 = unadjusted; 6 = holding periods (table4_holding_period_sweep.csv);
# 21 = 10 signal constructions (dsr_results_v2.csv) + 6 holding periods + 5 OFI definitions (pbo_results_v2.csv)
port = daily.groupby(level=0)[["pnl_gross", "pnl_net"]].sum().sort_index()
print(f"\nEQUAL-CAPITAL PORTFOLIO (walk-forward strategy): {port.index.min().date()} -> {port.index.max().date()}")
for col in ("pnl_gross", "pnl_net"):
    for nt in (1, 6, 21):
        n, sra, sk, exk, d = dsr(port[col], nt)
        print(f"  {col:9s} N_trials={nt:<4d} n={n}  SR_ann={sra:+.3f}  skew={sk:+.2f}  ex_kurt={exk:.2f}  DSR={d:.4f}")

print("\nPer ticker (daily, annualised):")
for tk, g in daily.groupby("ticker"):
    a, b = g["pnl_gross"], g["pnl_net"]
    print(f"  {tk:5s} n={len(g):4d}  gross SR={a.mean()/a.std()*np.sqrt(252):+.2f}  net SR={b.mean()/b.std()*np.sqrt(252):+.2f}")
