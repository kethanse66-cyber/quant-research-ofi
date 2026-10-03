import glob, os, io
import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"E:/quant-research-ofi"
R = os.path.join(ROOT, "reports"); F = os.path.join(ROOT, "features")
buf = io.StringIO()
def p(*a):
    print(*a); print(*a, file=buf)

# ===================================================== PART 1: every result file in the project
p("=" * 90); p("PART 1 — ALL RESULT FILES IN THE PROJECT"); p("=" * 90)
skip = (".venv", "__pycache__", "\\raw\\", "/raw/", "raw_parquet", "\\data\\", "/data/", "\\parquet\\", "/parquet/")
exts = ("*.csv", "*.parquet", "*.png", "*.txt", "*.json", "*.docx", "*.pdf", "*.html")
rows = []
for ext in exts:
    for f in glob.glob(os.path.join(ROOT, "**", ext), recursive=True):
        if any(s in f for s in skip) or f.endswith("_features.parquet"):
            continue
        st = os.stat(f)
        rows.append((os.path.relpath(f, ROOT), pd.Timestamp(st.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"), st.st_size))
inv = pd.DataFrame(rows, columns=["file", "last_saved", "bytes"]).sort_values("file")
p(inv.to_string(index=False))
p(f"\nTotal result files: {len(inv)}")

# ===================================================== PART 2: every paper number vs its saved file
p("\n" + "=" * 90); p("PART 2 — PAPER NUMBERS CHECKED AGAINST SAVED FILES"); p("=" * 90)
res = []
def check(section, claim, got, expected, tol=0.0015):
    try:
        ok = abs(float(got) - float(expected)) <= tol
        res.append((section, claim, f"{float(got):.4f}", expected, "OK" if ok else "MISMATCH"))
    except Exception as ex:
        res.append((section, claim, str(got), expected, "ERROR"))
def missing(section, claim, path):
    res.append((section, claim, "-", "-", f"FILE MISSING: {path}"))
def load(name, folder=R):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return None, path
    return (pd.read_csv(path) if name.endswith(".csv") else pd.read_parquet(path)), path

# Table 2 — regimes
df, path = load("regime_characterization.csv")
if df is None: missing("Table 2", "regime split", path)
else:
    s = df.set_index("regime")["n_bars"]; check("Table 2", "stressed % of bars", 100 * s["stressed"] / s.sum(), 72.9, 0.1)
df, path = load("regime_duration.csv")
if df is None: missing("Table 2", "durations", path)
else:
    d = df.set_index(df.columns[0])["avg_duration_minutes"]
    check("Table 2", "calm duration (min)", d["calm"], 0.72, 0.01); check("Table 2", "stressed duration (min)", d["stressed"], 1.92, 0.01)

# Table 3 — IC summary (mixed evaluation file)
df, path = load("SPY_ic_table1_mixed.csv")
if df is None: missing("Table 3", "IC table", path)
else:
    g = df.set_index(["horizon", "model"])
    for h, m, ic, t in [("fwd_ret_30s", "Ridge-Global", 0.0057, 3.44), ("fwd_ret_1m", "Ridge-Global", 0.0040, 2.11),
                        ("fwd_ret_10s", "Ridge-Global", 0.0031, 1.86), ("fwd_ret_5m", "Ridge-Global", 0.0028, 0.45),
                        ("fwd_ret_10m", "Ridge-Global", 0.0008, 0.06),
                        ("fwd_ret_5m", "Ridge-Calm-All", 0.0161, 2.26), ("fwd_ret_10m", "Ridge-Calm-All", 0.0148, 2.30),
                        ("fwd_ret_30s", "Ridge-Calm-All", -0.0066, -4.21)]:
        check("Table 3", f"{m} {h} IC", g.loc[(h, m), "mean_ic"], ic, 0.0002)
        check("Table 3", f"{m} {h} t", g.loc[(h, m), "tstat_nw"], t, 0.01)
df, path = load("SPY_ic_table2_matched.csv")
if df is None: missing("Table 3", "matched table", path)
else:
    g = df.set_index(["horizon", "model"])
    check("Table 3", "Ridge-Calm-Matched 10m IC", g.loc[("fwd_ret_10m", "Ridge-Calm-Matched"), "mean_ic"], -0.0303, 0.0002)
    check("Table 3", "Ridge-Calm-Matched 10m t", g.loc[("fwd_ret_10m", "Ridge-Calm-Matched"), "tstat_nw"], -2.81, 0.01)

# Section 5.3 / Table 5 — walk-forward
df, path = load("walk_forward_results.parquet", F)
if df is None: missing("5.3", "walk-forward", path)
else:
    if "horizon" in df.columns:
        df = df[df["horizon"].astype(str).str.contains("10m")]
    t = stats.ttest_1samp(df["ic"], 0)
    check("5.3", "mean IC", df["ic"].mean(), 0.0044, 0.0001); check("5.3", "IC t-stat", t.statistic, 3.26, 0.01)
    check("5.3", "IC p-value", t.pvalue, 0.0016, 0.0001)
    check("5.3", "mean gross Sharpe", df["sharpe_gross"].mean(), 0.981, 0.001)
    check("5.3", "mean net Sharpe", df["sharpe_net"].mean(), -1.726, 0.001)
    check("5.3", "survival %", 100 * df["survives"].astype(float).mean(), 35.7, 0.1)
    for col, tv, pv in (("ticker", 4.02, 0.0020), ("test_month", 2.26, 0.065)):
        gm = df.groupby(col)["ic"].mean(); tt = stats.ttest_1samp(gm, 0)
        check("5.3", f"cluster by {col} t", tt.statistic, tv, 0.01); check("5.3", f"cluster by {col} p", tt.pvalue, pv, 0.001)
    m = df.groupby("test_month")["sharpe_gross"].mean()
    check("5.3", "trend r", np.corrcoef(np.arange(len(m)), m.values)[0, 1], 0.84, 0.01)
    nv = df[df["ticker"] == "NVDA"]
    check("Table 5", "NVDA net Sharpe", nv["sharpe_net"].mean(), 0.140, 0.001)
    check("Table 5", "NVDA gross Sharpe", nv["sharpe_gross"].mean(), 1.308, 0.001)

# Table 4 — holding period sweep
df, path = load("table4_holding_period_sweep.csv")
if df is None: missing("Table 4", "sweep", path)
else:
    g = df.set_index("holding_period")
    check("Table 4", "30m net Sharpe", g.loc["30m", "sharpe_net"], 0.114, 0.001)
    check("Table 4", "30m gross Sharpe", g.loc["30m", "sharpe_gross"], 0.878, 0.001)
    res.append(("Table 4", "source", "-", "-", "NOTE: CSV was written from hardcoded values, not a saved backtest run"))

# 7.1 — DSR from daily walk-forward PnL
df, path = load("wf_daily_pnl.parquet", F)
if df is None: missing("7.1", "daily PnL for DSR", path)
else:
    df.index = pd.Index([pd.Timestamp(x).tz_convert("America/New_York").tz_localize(None) if pd.Timestamp(x).tzinfo
                         else pd.Timestamp(x) for x in df.index]).normalize()
    port = df.groupby(level=0)[["pnl_gross", "pnl_net"]].sum()
    def dsr(r, N):
        r = r.values; n = len(r); sr = r.mean() / r.std(ddof=1); sk = stats.skew(r); ku = stats.kurtosis(r, fisher=False)
        e = 0.5772156649
        sr0 = 0 if N == 1 else np.sqrt(1 / n) * ((1 - e) * stats.norm.ppf(1 - 1 / N) + e * stats.norm.ppf(1 - 1 / (N * np.e)))
        return sr * np.sqrt(252), stats.norm.cdf((sr - sr0) * np.sqrt(n - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)), n
    sg, d1, n = dsr(port["pnl_gross"], 1); _, d21, _ = dsr(port["pnl_gross"], 21); _, d6, _ = dsr(port["pnl_gross"], 6)
    sn, dn1, _ = dsr(port["pnl_net"], 1)
    check("7.1", "portfolio days", n, 145, 0); check("7.1", "gross Sharpe", sg, 2.75, 0.01); check("7.1", "net Sharpe", sn, -1.59, 0.01)
    check("7.1", "gross DSR N=1", d1, 0.995, 0.001); check("7.1", "gross DSR N=6", d6, 0.84, 0.005)
    check("7.1", "gross DSR N=21", d21, 0.58, 0.005); check("7.1", "net DSR N=1", dn1, 0.15, 0.005)

# 7.1 — PBO
for name, val in (("pbo_results.csv", 0.3143), ("pbo_results_v2.csv", 0.80), ("pbo_results_v3.csv", 0.1429)):
    df, path = load(name)
    if df is None: missing("7.1", f"PBO {name}", path)
    else: check("7.1", f"PBO ({name})", df["pbo"].iloc[0], val, 0.0001)
df, path = load("dsr_results_v2.csv")
if df is None: missing("7.1", "10 signal variants", path)
else: check("7.1", "signal variants documented", len(df), 10, 0)
df, path = load("table4_holding_period_sweep.csv")
if df is not None: check("7.1", "holding periods documented", len(df), 6, 0)
df, path = load("pbo_results_v2.csv")
if df is not None: check("7.1", "OFI variants documented", df["n_strategies"].iloc[0], 5, 0)

# 7.1 — OOS
df, path = load("oos_summary.csv")
if df is None: missing("7.1", "OOS", path)
else:
    check("7.1", "OOS IC", df["oos_ic"].iloc[0], 0.0022, 0.0001); check("7.1", "OOS gross Sharpe", df["oos_sharpe"].iloc[0], 2.491, 0.001)
    check("7.1", "OOS net Sharpe", df["oos_net"].iloc[0], 0.246, 0.001); check("7.1", "Welch p", df["pval"].iloc[0], 0.30, 0.01)

# 7.4 — LOBSTER / MBO
df, path = load("lobster_comparison_v2.csv")
if df is None: missing("7.4", "L1 vs L2", path)
else:
    g = df.set_index("horizon")
    check("7.4", "5m IC L1", g.loc["fwd_ret_5m", "ic_l1"], -0.0026, 0.0001); check("7.4", "5m IC L2", g.loc["fwd_ret_5m", "ic_l2"], -0.0046, 0.0001)
    check("7.4", "5m p L1", g.loc["fwd_ret_5m", "pval_l1"], 0.43, 0.01); check("7.4", "5m p L2", g.loc["fwd_ret_5m", "pval_l2"], 0.16, 0.01)

# 7.2 — capacity (separate 1-min backtest)
df, path = load("capacity_analysis.csv")
if df is None: missing("7.2", "capacity", path)
else: check("7.2", "tickers with negative net Sharpe", (df["net_sharpe"] < 0).sum(), 12, 0)

# 8 — live validation: list candidate files (no fixed name known)
live = [f for f in glob.glob(os.path.join(ROOT, "**", "*.csv"), recursive=True)
        if "live" in f.lower() and ".venv" not in f]
res.append(("8", "live validation files found", str(len(live)), "-", "; ".join(os.path.relpath(x, ROOT) for x in live[:8]) or "NONE FOUND"))

out = pd.DataFrame(res, columns=["section", "claim", "from_file", "paper", "status"])
p(out.to_string(index=False))
p(f"\nOK: {(out.status == 'OK').sum()}   MISMATCH: {(out.status == 'MISMATCH').sum()}   "
  f"MISSING/ERROR: {out.status.str.contains('MISSING|ERROR').sum()}")

with open(os.path.join(R, "results_audit.txt"), "w", encoding="utf-8") as fh:
    fh.write(buf.getvalue())
print(f"\nSaved to {os.path.join(R, 'results_audit.txt')}")
