# performance_reporter.py
# Phase 6 — Full Tearsheet Generator
# Inputs:
#   walk_forward_results.parquet  — fold-level metrics (all 12 tickers)
#   SPY_backtest_pnl_final.parquet — bar-level PnL for SPY
#   SPY_features_with_regimes.parquet — regime labels for SPY
# Output:
#   reports/tearsheet.html — professional HTML tearsheet

import pandas as pd
import numpy as np
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
FEAT_DIR    = "E:/quant-research-ofi/features"
REPORT_DIR  = "E:/quant-research-ofi/reports"
OUTPUT_HTML = f"{REPORT_DIR}/tearsheet.html"

WF_PATH     = f"{FEAT_DIR}/walk_forward_results.parquet"
PNL_PATH    = f"{FEAT_DIR}/SPY_backtest_pnl_final.parquet"
REG_PATH    = f"{FEAT_DIR}/SPY_features_with_regimes.parquet"

os.makedirs(REPORT_DIR, exist_ok=True)

# ── LOAD ──────────────────────────────────────────────────────────────────────
wf  = pd.read_parquet(WF_PATH)
pnl = pd.read_parquet(PNL_PATH)
reg = pd.read_parquet(REG_PATH)

# ── 1. EQUITY CURVE FROM BAR-LEVEL PNL (SPY) ─────────────────────────────────
pnl.index = pd.to_datetime(pnl.index)
pnl = pnl.sort_index()

equity_gross = pnl['pnl_gross'].cumsum()
equity_net   = pnl['pnl_net'].cumsum()

# Resample to daily for cleaner chart
daily_gross = pnl['pnl_gross'].resample('D').sum().cumsum()
daily_net   = pnl['pnl_net'].resample('D').sum().cumsum()
daily_gross = daily_gross[daily_gross != 0].dropna()
daily_net   = daily_net[daily_net != 0].dropna()

# ── 2. DRAWDOWN FROM BAR-LEVEL PNL (SPY) ─────────────────────────────────────
def compute_drawdown(equity_series):
    roll_max = equity_series.cummax()
    dd = equity_series - roll_max
    return dd

dd_gross = compute_drawdown(daily_gross)
dd_net   = compute_drawdown(daily_net)

max_dd_gross = dd_gross.min()
max_dd_net   = dd_net.min()

# ── 3. ROLLING SHARPE FROM FOLD METRICS (ALL TICKERS) ─────────────────────────
# Use cross-sectional mean Sharpe per month as proxy for rolling Sharpe
monthly_sharpe = wf.groupby('test_month')['sharpe_gross'].mean().reset_index()
monthly_sharpe.columns = ['month', 'sharpe_gross']
monthly_net    = wf.groupby('test_month')['sharpe_net'].mean().reset_index()
monthly_sharpe['sharpe_net'] = monthly_net['sharpe_net']

# ── 4. IC OVER TIME (ALL TICKERS) ─────────────────────────────────────────────
monthly_ic = wf.groupby('test_month')['ic'].mean().reset_index()
monthly_ic.columns = ['month', 'mean_ic']

# ── 5. REGIME BREAKDOWN (SPY) ────────────────────────────────────────────────
reg.index = pd.to_datetime(reg.index)
reg = reg.dropna(subset=['regime'])
reg['regime'] = reg['regime'].astype(int)

regime_map   = {0: 'Calm', 1: 'Normal', 2: 'Stressed'}
regime_counts = reg['regime'].value_counts().sort_index()
regime_pct    = (regime_counts / regime_counts.sum() * 100).round(1)

regime_stats = reg.groupby('regime').agg(
    mean_vol   = ('realized_vol',    'mean'),
    mean_spread= ('spread',          'mean'),
    mean_qi    = ('queue_imbalance', 'mean'),
    count      = ('regime',          'count'),
).round(4)
regime_stats.index = [regime_map.get(i, str(i)) for i in regime_stats.index]
regime_stats['pct'] = regime_pct.values

# ── 6. SUMMARY TABLE ──────────────────────────────────────────────────────────
total_gross = pnl['pnl_gross'].sum()
total_net   = pnl['pnl_net'].sum()
total_cost  = pnl['total_cost'].sum()

spy_wf      = wf[wf['ticker'] == 'SPY']
mean_ic     = wf['ic'].mean()
ic_tstat    = wf['ic'].mean() / (wf['ic'].std() / np.sqrt(len(wf)))
surv_rate   = wf['survives'].mean()

agg_gross_sh = wf['sharpe_gross'].mean()
agg_net_sh   = wf['sharpe_net'].mean()
agg_dd       = wf['max_dd'].mean()
agg_turn     = wf['turnover_bar'].mean()

oos_wf       = wf[wf['test_month'].isin(['2025-03', '2025-04'])]
oos_ic        = oos_wf['ic'].mean()
oos_sh_gross  = oos_wf['sharpe_gross'].mean()
oos_sh_net    = oos_wf['sharpe_net'].mean()

# ── BUILD HTML ────────────────────────────────────────────────────────────────
def fmt(v, decimals=3):
    return f"{v:.{decimals}f}"

def color(v, good_positive=True):
    if good_positive:
        c = "#2ecc71" if v > 0 else "#e74c3c"
    else:
        c = "#e74c3c" if v > 0 else "#2ecc71"
    return f'<span style="color:{c};font-weight:bold">{fmt(v)}</span>'

def make_sparkline_svg(values, width=200, height=50, color="#3498db"):
    values = list(values)
    if len(values) < 2:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    pts = []
    for i, v in enumerate(values):
        x = i / (len(values) - 1) * width
        y = height - (v - mn) / rng * height
        pts.append(f"{x:.1f},{y:.1f}")
    path = " ".join(pts)
    zero_y = height - (0 - mn) / rng * height
    zero_y = max(0, min(height, zero_y))
    return f'''<svg width="{width}" height="{height}" style="display:block">
      <line x1="0" y1="{zero_y:.1f}" x2="{width}" y2="{zero_y:.1f}"
            stroke="#555" stroke-width="0.5" stroke-dasharray="3,3"/>
      <polyline points="{path}" fill="none" stroke="{color}" stroke-width="1.8"/>
    </svg>'''

equity_spark  = make_sparkline_svg(daily_net.values,        color="#2ecc71")
dd_spark      = make_sparkline_svg(dd_net.values,           color="#e74c3c")
sharpe_spark  = make_sparkline_svg(monthly_sharpe['sharpe_net'].values, color="#3498db")
ic_spark      = make_sparkline_svg(monthly_ic['mean_ic'].values,        color="#9b59b6")

regime_rows = ""
for rname, row in regime_stats.iterrows():
    regime_rows += f"""
    <tr>
      <td>{rname}</td>
      <td>{row['count']}</td>
      <td>{row['pct']:.1f}%</td>
      <td>{row['mean_vol']:.4f}</td>
      <td>{row['mean_spread']:.4f}</td>
      <td>{row['mean_qi']:.4f}</td>
    </tr>"""

monthly_rows = ""
for _, row in monthly_sharpe.iterrows():
    ic_val = monthly_ic[monthly_ic['month'] == row['month']]['mean_ic'].values
    ic_str = f"{ic_val[0]:+.4f}" if len(ic_val) else "—"
    surv   = wf[wf['test_month'] == row['month']]['survives'].mean()
    monthly_rows += f"""
    <tr>
      <td>{row['month']}</td>
      <td style="color:{'#2ecc71' if row['sharpe_gross']>0 else '#e74c3c'}">{row['sharpe_gross']:+.3f}</td>
      <td style="color:{'#2ecc71' if row['sharpe_net']>0 else '#e74c3c'}">{row['sharpe_net']:+.3f}</td>
      <td>{ic_str}</td>
      <td>{surv:.0%}</td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>OFI Strategy Tearsheet</title>
<style>
  body {{ font-family: 'Courier New', monospace; background: #0f0f0f; color: #e0e0e0;
          margin: 0; padding: 20px; font-size: 13px; }}
  h1   {{ color: #3498db; border-bottom: 1px solid #333; padding-bottom: 8px; font-size: 18px; }}
  h2   {{ color: #95a5a6; font-size: 13px; text-transform: uppercase;
          letter-spacing: 2px; margin-top: 30px; margin-bottom: 8px; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }}
  .card {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 4px;
           padding: 14px; }}
  .card .label {{ color: #888; font-size: 11px; margin-bottom: 4px; }}
  .card .value {{ font-size: 20px; font-weight: bold; }}
  .card .sub   {{ color: #666; font-size: 10px; margin-top: 4px; }}
  .pos {{ color: #2ecc71; }} .neg {{ color: #e74c3c; }} .neu {{ color: #f39c12; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
  th    {{ background: #1a1a1a; color: #888; padding: 8px 12px; text-align: left;
           font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }}
  td    {{ padding: 7px 12px; border-bottom: 1px solid #1e1e1e; font-size: 12px; }}
  tr:hover td {{ background: #1a1a1a; }}
  .warn {{ background: #2a1a00; border: 1px solid #f39c12; border-radius: 4px;
           padding: 10px 14px; margin-bottom: 16px; color: #f39c12; font-size: 11px; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px; }}
  .chart-box {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 4px; padding: 14px; }}
  .chart-box .chart-title {{ color: #888; font-size: 11px; margin-bottom: 8px; text-transform: uppercase; }}
</style>
</head>
<body>

<h1>OFI Cross-Asset Strategy — Performance Tearsheet</h1>
<p style="color:#555;font-size:11px">
  Universe: SPY QQQ IWM XLF XLK XLE XLV AAPL JPM NVDA ES1! TLT &nbsp;|&nbsp;
  Period: Oct 2024 – Apr 2025 &nbsp;|&nbsp;
  Walk-Forward: 6-month expanding train, 1-month test &nbsp;|&nbsp;
  Signal: OFI-based Ridge regression
</p>

<div class="warn">
  ⚠ OOS window is only 2 months (Mar–Apr 2025) and may not be representative.
  All OOS metrics should be interpreted cautiously — 2 months is insufficient for robust inference.
</div>

<h2>Key Metrics</h2>
<div class="grid">
  <div class="card">
    <div class="label">Mean IC (all folds)</div>
    <div class="value {'pos' if mean_ic > 0 else 'neg'}">{mean_ic:+.4f}</div>
    <div class="sub">t-stat: {ic_tstat:.2f} (approx — folds not independent) &nbsp;|&nbsp; OOS IC: {oos_ic:+.4f}</div>
  </div>
  <div class="card">
    <div class="label">Mean Gross Sharpe</div>
    <div class="value {'pos' if agg_gross_sh > 0 else 'neg'}">{agg_gross_sh:+.3f}</div>
    <div class="sub">OOS: {oos_sh_gross:+.3f}</div>
  </div>
  <div class="card">
    <div class="label">Mean Net Sharpe</div>
    <div class="value {'pos' if agg_net_sh > 0 else 'neg'}">{agg_net_sh:+.3f}</div>
    <div class="sub">OOS: {oos_sh_net:+.3f}</div>
  </div>
  <div class="card">
    <div class="label">Survival Rate</div>
    <div class="value neu">{surv_rate:.0%}</div>
    <div class="sub">Folds where net Sharpe > 0</div>
  </div>
  <div class="card">
    <div class="label">SPY Total Gross PnL</div>
    <div class="value {'pos' if total_gross > 0 else 'neg'}">{total_gross:+.2f}</div>
    <div class="sub">Net: {total_net:+.2f}</div>
  </div>
  <div class="card">
    <div class="label">SPY Max Drawdown (net)</div>
    <div class="value neg">{max_dd_net:.4f}</div>
    <div class="sub">Gross: {max_dd_gross:.4f}</div>
  </div>
  <div class="card">
    <div class="label">Mean Turnover / Bar</div>
    <div class="value neu">{agg_turn:.4f}</div>
    <div class="sub">Fraction of portfolio traded</div>
  </div>
  <div class="card">
    <div class="label">Mean Max Drawdown</div>
    <div class="value neg">{agg_dd:.4f}</div>
    <div class="sub">Avg across folds and tickers</div>
  </div>
</div>

<h2>Sparklines</h2>
<div class="charts">
  <div class="chart-box">
    <div class="chart-title">SPY Cumulative Net PnL</div>
    {equity_spark}
  </div>
  <div class="chart-box">
    <div class="chart-title">SPY Net Drawdown</div>
    {dd_spark}
  </div>
  <div class="chart-box">
    <div class="chart-title">Rolling Net Sharpe (cross-sectional mean)</div>
    {sharpe_spark}
  </div>
  <div class="chart-box">
    <div class="chart-title">Mean IC Over Time</div>
    {ic_spark}
  </div>
</div>

<h2>Monthly Breakdown (All Tickers)</h2>
<table>
  <tr>
    <th>Month</th>
    <th>Gross Sharpe</th>
    <th>Net Sharpe</th>
    <th>Mean IC</th>
    <th>Survival</th>
  </tr>
  {monthly_rows}
</table>

<h2>Regime Characterisation (SPY)</h2>
<table>
  <tr>
    <th>Regime</th>
    <th>Bars</th>
    <th>% Time</th>
    <th>Mean RealVol</th>
    <th>Mean Spread</th>
    <th>Mean QueueImbal</th>
  </tr>
  {regime_rows}
</table>

<h2>Factor Attribution</h2>
<p style="color:#666;font-size:11px">
  Run factor_model.py (Jun 13) for full Fama-French 3-factor regression.
  Placeholder: strategy returns are fold-level net Sharpe series — bar-level
  returns needed for FF3 regression. To be completed tomorrow.
</p>

<h2>Honest Assessment</h2>
<table>
  <tr><th>Item</th><th>Finding</th><th>Interpretation</th></tr>
  <tr>
    <td>OOS IC</td>
    <td class="{'pos' if oos_ic > 0 else 'neg'}">{oos_ic:+.4f}</td>
    <td>Signal direction correct OOS</td>
  </tr>
  <tr>
    <td>OOS Gross Sharpe</td>
    <td style="color:#f39c12">{oos_sh_gross:+.3f}</td>
    <td>Upward biased — Apr 2025 tariff shock in OOS window</td>
  </tr>
  <tr>
    <td>OOS Net Sharpe</td>
    <td class="{'pos' if oos_sh_net > 0 else 'neg'}">{oos_sh_net:+.3f}</td>
    <td>Positive during OOS window — sensitive to costs and regime</td>
  </tr>
  <tr>
    <td>IS Months Oct–Nov</td>
    <td class="neg">Negative</td>
    <td>Low-volatility regime — signal weak, costs dominate</td>
  </tr>
  <tr>
    <td>OOS Window</td>
    <td style="color:#f39c12">2 months</td>
    <td>Too small for robust inference — interpret cautiously</td>
  </tr>
</table>

<p style="color:#333;font-size:10px;margin-top:30px">
  Generated by performance_reporter.py &nbsp;|&nbsp;
  quant-research-ofi &nbsp;|&nbsp;
  Phase 6 Backtesting
</p>

</body>
</html>"""

with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)

print("=" * 60)
print("TEARSHEET COMPLETE")
print("=" * 60)
print(f"  Output : {OUTPUT_HTML}")
print()
print(f"  Mean IC          : {mean_ic:+.4f}  (t-stat: {ic_tstat:.2f})")
print(f"  Mean Gross Sharpe: {agg_gross_sh:+.3f}")
print(f"  Mean Net Sharpe  : {agg_net_sh:+.3f}")
print(f"  Survival Rate    : {surv_rate:.0%}")
print(f"  SPY Max DD (net) : {max_dd_net:.4f}")
print(f"  OOS IC           : {oos_ic:+.4f}")
print(f"  OOS Net Sharpe   : {oos_sh_net:+.3f}")
print()
print("  Open tearsheet.html in any browser.")
