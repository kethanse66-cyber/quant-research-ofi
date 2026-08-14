# run_backtest_spy.py
# Phase 6 Backtest — June 4
# First complete SPY backtest: position_sizer + transaction_costs + performance metrics
#

import numpy as np
import pandas as pd
import sys
import os

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from position_sizer import compute_rolling_vol, vol_target_position, apply_position_cap, position_summary
from transaction_costs import apply_transaction_costs, cost_summary

PORTFOLIO_VALUE = 1_000_000
ANN_FACTOR = np.sqrt(252 * 6.5 * 360)  # annualization for 10-second bars

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading SPY features...")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "features", "SPY_features.parquet")
df = pd.read_parquet(FEATURES_PATH)
df = df[df['ticker'] == 'SPY'].copy()
df = df.sort_index()

print(f"  Rows loaded : {len(df):,}")
print(f"  Date range  : {df.index[0]}  ->  {df.index[-1]}")

# ── 2. PRICE AND RETURNS ──────────────────────────────────────────────────────
price = df['microprice'].ffill().bfill()
log_returns = np.log(price / price.shift(1)).fillna(0)

# ── 3. SIGNAL ─────────────────────────────────────────────────────────────────
# Fix 3: signal is -ofi_norm
# ofi_norm is negatively correlated with fwd_ret_10m
# confirmed by correlation matrix and gross Sharpe flipping from -5.6 to +1.9
signal = -df['ofi_norm'].fillna(0)

# ── 4. GROSS SIGNAL DIAGNOSTIC ───────────────────────────────────────────────
# Check raw signal edge BEFORE position sizing or costs
# If gross Sharpe < 0 here — wrong direction or wrong horizon — stop immediately
forward_ret = df['fwd_ret_10m'].fillna(0)
raw_pnl = signal.shift(1).fillna(0) * forward_ret * price
gross_sharpe_raw = (raw_pnl.mean() / raw_pnl.std()) * ANN_FACTOR if raw_pnl.std() > 0 else np.nan

print("\n── GROSS SIGNAL DIAGNOSTIC (before sizing and costs) ──")
print(f"  Signal                : -ofi_norm")
print(f"  Forward return target : fwd_ret_10m")
print(f"  Raw gross PnL ($)     : {raw_pnl.sum():,.2f}")
print(f"  Raw gross Sharpe      : {gross_sharpe_raw:.3f}")
print(f"  Signal has edge       : {raw_pnl.sum() > 0}")

# ── 5. ROLLING VOL ────────────────────────────────────────────────────────────
# vol clipped to [0.05, 1.0] — see position_sizer.py Fix 5+6
# shift(1) applied inside — no lookahead
rolling_vol = compute_rolling_vol(
    returns=log_returns,
    window=120,
    annualize=True,
    bars_per_day=2340
)

# ── 6. SIGNAL NORMALIZATION ───────────────────────────────────────────────────
# Normalize over 11700 bars = 5 trading days
# Single normalization here — internal normalization in vol_target_position disabled
# Fix 7: double normalization was causing position std=170 and $15B turnover
NORM_WINDOW = 11700
signal_norm = signal / signal.rolling(NORM_WINDOW).std().shift(1).clip(lower=1e-6)
signal_norm = signal_norm.fillna(0)

# ── 7. POSITION SIZING ────────────────────────────────────────────────────────
positions_raw = vol_target_position(
    signal=signal_norm,
    rolling_vol=rolling_vol,
    target_daily_vol=0.0075,  # 0.75% — between 0.5% and 1%, balances gross vs cost
    portfolio_value=PORTFOLIO_VALUE,
    price=price
)

# Cap at 10% of portfolio = $100,000 max dollar exposure
positions = apply_position_cap(
    positions=positions_raw,
    price=price,
    portfolio_value=PORTFOLIO_VALUE,
    max_position_pct=0.10
)

# ── 8. HOLDING PERIOD ─────────────────────────────────────────────────────────
# Fix 5: only update position every 60 bars = every 10 minutes
# Without this: position recalculates every 10s → 2340 updates/day → $61M turnover
# With this: position updates every 10min → 39 updates/day → ~$1M turnover
# Matches fwd_ret_10m signal horizon — position held for as long as signal is valid
HOLDING_BARS = 180  # 30 minutes — cost trend: 10min=237%, 20min=150%, target <100%
positions = positions.where(
    pd.Series(range(len(positions)), index=positions.index) % HOLDING_BARS == 0,
    other=np.nan
).ffill()

# shift(1): enter at bar T+1 after signal at bar T — no lookahead
positions = positions.shift(1).fillna(0)

print("\n")
position_summary(positions, price, portfolio_value=PORTFOLIO_VALUE)

# ── 9. GROSS PNL ──────────────────────────────────────────────────────────────
# Fix 1: shares * price_change = dollar PnL — correct units
# positions already shifted — shift(0) makes timing explicit for any reviewer
gross_pnl = positions.shift(0).fillna(0) * price.diff().fillna(0)

# ── 10. ADV ───────────────────────────────────────────────────────────────────
adv = pd.Series(80_000_000, index=df.index)

# ── 11. PRICES DATAFRAME FOR COST MODULE ─────────────────────────────────────
prices_df = pd.DataFrame({
    'mid': price,
    'spread': df['spread'].ffill().bfill(),
    'realized_vol': rolling_vol.ffill().bfill()
}, index=df.index)

# ── 12. TRANSACTION COSTS ────────────────────────────────────────────────────
cost_df = apply_transaction_costs(
    pnl_gross=gross_pnl,
    positions=positions,
    prices=prices_df,
    adv=adv,
    commission_bps=0.1
)

print("\n")
cost_summary(cost_df)

# ── 13. PORTFOLIO RETURNS AND EQUITY CURVE ───────────────────────────────────
# Fix 2: pnl is dollars — divide by portfolio_value before cumprod
pnl_net = cost_df['pnl_net'].dropna()
pnl_gross_series = cost_df['pnl_gross'].dropna()

returns_net = pnl_net / PORTFOLIO_VALUE
returns_gross = pnl_gross_series / PORTFOLIO_VALUE

equity_net = (1 + returns_net).cumprod()
equity_gross = (1 + returns_gross).cumprod()

# ── 14. PERFORMANCE METRICS ──────────────────────────────────────────────────
sharpe_net   = (returns_net.mean()   / returns_net.std())   * ANN_FACTOR if returns_net.std()   > 0 else np.nan
sharpe_gross = (returns_gross.mean() / returns_gross.std()) * ANN_FACTOR if returns_gross.std() > 0 else np.nan

rolling_max = equity_net.cummax()
drawdown    = (equity_net / rolling_max) - 1
max_dd      = drawdown.min()

n_bars        = len(returns_net)
bars_per_year = 252 * 2340
ann_return    = (equity_net.iloc[-1] ** (bars_per_year / n_bars)) - 1
calmar        = ann_return / abs(max_dd) if max_dd != 0 else np.nan

dollar_turnover_per_bar = (positions.diff().abs() * price).fillna(0)
portfolio_turnover      = (dollar_turnover_per_bar / PORTFOLIO_VALUE).mean()

win_rate = (returns_net > 0).mean()

# ── 15. PRINT RESULTS ────────────────────────────────────────────────────────
print("\n")
print("=" * 50)
print("FULL BACKTEST RESULTS — SPY  (final)")
print("=" * 50)
print(f"Signal            : -ofi_norm  (flipped, 5-day normalized)")
print(f"Target horizon    : fwd_ret_10m")
print(f"Holding period    : 180 bars = 30 minutes")
print(f"Bars              : {n_bars:,}  ({n_bars/2340:.1f} trading days)")
print(f"Date range        : {pnl_net.index[0]}  ->  {pnl_net.index[-1]}")
print("-" * 50)
print(f"Sharpe (gross)    : {sharpe_gross:.3f}")
print(f"Sharpe (net)      : {sharpe_net:.3f}")
print(f"Annualized return : {ann_return*100:.2f}%")
print(f"Max drawdown      : {max_dd:.2%}")
print(f"Calmar ratio      : {calmar:.3f}")
print(f"Win rate          : {win_rate:.2%}")
print(f"Avg turnover/bar  : {portfolio_turnover:.6f}  (fraction of portfolio)")
print(f"Final equity      : {equity_net.iloc[-1]:.4f}")
print("=" * 50)

# ── 16. HONEST ASSESSMENT ────────────────────────────────────────────────────
gross_total = cost_df['pnl_gross'].sum()
net_total   = cost_df['pnl_net'].sum()
cost_drag   = gross_total - net_total

print("\nHONEST ASSESSMENT:")
print(f"  Gross PnL ($)      : {gross_total:,.2f}")
print(f"  Net PnL ($)        : {net_total:,.2f}")
print(f"  Cost drag ($)      : {cost_drag:,.2f}")
if abs(gross_total) > 1e-6:
    print(f"  Costs ate          : {cost_drag/gross_total*100:.1f}% of gross PnL")
print(f"  Signal survives    : {net_total > 0}")

# ── 17. SAVE ─────────────────────────────────────────────────────────────────
out_path = os.path.join(PROJECT_ROOT, "features", "SPY_backtest_pnl_final.parquet")
cost_df.to_parquet(out_path)
print(f"\nPnL series saved: {out_path}")
print("Push to GitHub after reviewing numbers.")
