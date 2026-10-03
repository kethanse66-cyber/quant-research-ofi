# wf_daily_dsr.py
# Re-runs walk_forward_backtest.py unchanged, but ALSO saves daily PnL per fold
# so a valid Deflated Sharpe Ratio can be computed. Does NOT overwrite any existing file.
# Expanding window walk-forward backtest across all 12 tickers
#

import numpy as np
import pandas as pd
import sys
import os
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from position_sizer import compute_rolling_vol, vol_target_position, apply_position_cap
from transaction_costs import apply_transaction_costs

PORTFOLIO_VALUE    = 1_000_000
ANN_FACTOR         = np.sqrt(252 * 6.5 * 360)
HOLDING_BARS       = 180       # rebalance every 30 min — not true entry/exit tracking
NORM_WINDOW        = 11700     # 5 trading days normalization window
TRAIN_MONTHS       = 6         # minimum train period before first test fold
SIGNAL_THRESHOLD   = 0.5       # only trade when |signal_norm| > threshold
                                # filters weak signals, controls turnover

TICKERS = ['SPY','QQQ','IWM','XLF','XLK','XLE','XLV','AAPL','JPM','NVDA','TLT','ES1!']

# Fix 3: ADV per asset — market impact is meaningless with uniform 80M
# ETFs: ~80M shares/day | Single names: ~30M | ES1!: ~1.2M contracts/day
ADV_MAP = {
    'SPY' : 80_000_000,
    'QQQ' : 50_000_000,
    'IWM' : 30_000_000,
    'XLF' : 40_000_000,
    'XLK' : 15_000_000,
    'XLE' : 15_000_000,
    'XLV' : 10_000_000,
    'AAPL': 60_000_000,
    'JPM' : 10_000_000,
    'NVDA': 40_000_000,
    'TLT' : 10_000_000,
    'ES1!':  1_200_000,
}


def run_backtest_slice(df_slice: pd.DataFrame,
                       ticker: str,
                       train_signal_std: float) -> dict:
    """
    Run backtest on one test fold using normalization fitted on train data.

    Parameters
    ----------
    df_slice         : test period DataFrame
    ticker           : ticker string for ADV lookup
    train_signal_std : signal std fitted on TRAIN data only — no leakage

    Returns
    -------
    dict of performance metrics or None if insufficient data
    """
    if len(df_slice) < 500:
        return None

    price       = df_slice['microprice'].ffill().bfill()
    log_returns = np.log(price / price.shift(1)).fillna(0)
    signal      = -df_slice['ofi_norm'].fillna(0)
    forward_ret = df_slice['fwd_ret_10m'].fillna(0)

    # IC — rank correlation signal vs forward return
    valid = signal.shift(1).notna() & forward_ret.notna()
    if valid.sum() < 100:
        return None
    ic, ic_pval = stats.spearmanr(
        signal.shift(1).fillna(0)[valid],
        forward_ret[valid]
    )

    # Rolling vol
    rolling_vol = compute_rolling_vol(
        log_returns, window=120, annualize=True, bars_per_day=2340
    )

    # Fix 1: use train_signal_std for normalization — fitted on train, applied to test
    # previously: signal.rolling(NORM_WINDOW).std() on test data = leakage
    if train_signal_std < 1e-6:
        return None
    signal_norm = (signal / train_signal_std).fillna(0)

    # Fix 4: signal threshold — zero out weak signals to control turnover
    # only hold position when |signal_norm| > SIGNAL_THRESHOLD
    signal_filtered = signal_norm.where(signal_norm.abs() > SIGNAL_THRESHOLD, other=0.0)

    # Position sizing
    positions_raw = vol_target_position(
        signal_filtered, rolling_vol, 0.0075, PORTFOLIO_VALUE, price
    )
    positions = apply_position_cap(positions_raw, price, PORTFOLIO_VALUE, 0.10)

    # Fix 2: periodic rebalancing every 30 min (not true entry/exit holding period)
    # positions updated at fixed intervals — acceptable for research-grade backtest
    positions = positions.where(
        pd.Series(range(len(positions)), index=positions.index) % HOLDING_BARS == 0,
        other=np.nan
    ).ffill()
    positions = positions.shift(1).fillna(0)

    # Fix 5: shift(0) removed — positions already shifted above
    gross_pnl = positions * price.diff().fillna(0)

    # Fix 3: per-asset ADV
    adv = pd.Series(ADV_MAP.get(ticker, 10_000_000), index=df_slice.index)

    prices_df = pd.DataFrame({
        'mid'         : price,
        'spread'      : df_slice['spread'].ffill().bfill(),
        'realized_vol': rolling_vol.ffill().bfill()
    }, index=df_slice.index)

    cost_df = apply_transaction_costs(
        gross_pnl, positions, prices_df, adv, commission_bps=0.1
    )

    pnl_net   = cost_df['pnl_net'].dropna()
    pnl_gross = cost_df['pnl_gross'].dropna()
    if len(pnl_net) < 100:
        return None

    ret_net   = pnl_net / PORTFOLIO_VALUE
    ret_gross = pnl_gross / PORTFOLIO_VALUE

    sharpe_net   = (ret_net.mean()   / ret_net.std())   * ANN_FACTOR if ret_net.std()   > 0 else np.nan
    sharpe_gross = (ret_gross.mean() / ret_gross.std()) * ANN_FACTOR if ret_gross.std() > 0 else np.nan

    equity = (1 + ret_net).cumprod()
    max_dd = ((equity / equity.cummax()) - 1).min()

    gross_total = cost_df['pnl_gross'].sum()
    net_total   = cost_df['pnl_net'].sum()
    cost_pct    = (gross_total - net_total) / gross_total * 100 if abs(gross_total) > 1e-6 else np.nan

    turnover = (positions.diff().abs() * price).fillna(0).mean() / PORTFOLIO_VALUE

    DAILY.append(pd.DataFrame({
        'ticker'   : ticker,
        'pnl_gross': cost_df['pnl_gross'].resample('D').sum(),
        'pnl_net'  : cost_df['pnl_net'].resample('D').sum(),
    }))

    return {
        'ic'          : ic,
        'ic_pval'     : ic_pval,
        'sharpe_gross': sharpe_gross,
        'sharpe_net'  : sharpe_net,
        'net_pnl'     : net_total,
        'cost_pct'    : cost_pct,
        'max_dd'      : max_dd,
        'turnover_bar': turnover,
        'n_bars'      : len(df_slice),
        'survives'    : net_total > 0,
    }


DAILY = []

# ── Main loop ─────────────────────────────────────────────────────────────────
all_results = []

for ticker in TICKERS:
    fname = os.path.join(PROJECT_ROOT, "features", f"{ticker}_features.parquet")
    if not os.path.exists(fname):
        print(f"  MISSING: {fname}")
        continue

    df = pd.read_parquet(fname)
    df = df[df['ticker'] == ticker].copy()

    # ensure DatetimeIndex — some tickers store timestamp as column not index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'ts' in df.columns:
            df = df.set_index('ts')
        elif 'date' in df.columns:
            df = df.set_index('date')
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    if len(df) < 1000:
        print(f"  SKIP {ticker}: too few rows ({len(df)})")
        continue

    df['month'] = df.index.to_period('M')
    months      = sorted(df['month'].unique())

    if len(months) < TRAIN_MONTHS + 1:
        print(f"  SKIP {ticker}: not enough months ({len(months)})")
        continue

    print(f"\nProcessing {ticker}  ({len(df):,} bars, {len(months)} months)")

    for i in range(TRAIN_MONTHS, len(months)):
        train_months = months[:i]
        test_month   = months[i]

        df_train = df[df['month'].isin(train_months)]
        df_test  = df[df['month'] == test_month]

        # Fix 1: fit normalization std on TRAIN data only
        # rolling(NORM_WINDOW).std() on train, take last value as the std estimate
        # this is then passed into run_backtest_slice and applied to test signal
        train_sig = -df_train['ofi_norm'].fillna(0)
        rolling_std_train = train_sig.rolling(NORM_WINDOW).std().dropna()
        if len(rolling_std_train) == 0:
            continue
        train_signal_std = rolling_std_train.iloc[-1]

        result = run_backtest_slice(df_test, ticker, train_signal_std)
        if result is None:
            continue

        result['ticker']      = ticker
        result['test_month']  = str(test_month)
        result['train_months']= len(train_months)
        all_results.append(result)

        print(f"  {str(test_month)}: IC={result['ic']:+.4f}  "
              f"Sharpe(G)={result['sharpe_gross']:+.3f}  "
              f"Sharpe(N)={result['sharpe_net']:+.3f}  "
              f"Turnover={result['turnover_bar']:.5f}  "
              f"{'✓' if result['survives'] else '✗'}")

# ── Aggregate results ─────────────────────────────────────────────────────────
if not all_results:
    print("\nNo results — check data files")
    sys.exit(1)

res_df = pd.DataFrame(all_results)

print("\n")
print("=" * 95)
print("WALK-FORWARD SUMMARY — ALL TICKERS")
print("=" * 95)
print(f"{'Ticker':<8} {'Folds':>6} {'Mean IC':>9} {'ICIR':>7} "
      f"{'Sharpe(G)':>10} {'Sharpe(N)':>10} {'Net PnL($)':>11} {'Survive%':>9}")
print("-" * 95)

ticker_summary = []
for ticker in TICKERS:
    t = res_df[res_df['ticker'] == ticker]
    if len(t) == 0:
        continue

    mean_ic     = t['ic'].mean()
    icir        = t['ic'].mean() / t['ic'].std() if t['ic'].std() > 0 else np.nan
    mean_sg     = t['sharpe_gross'].mean()
    mean_sn     = t['sharpe_net'].mean()
    total_pnl   = t['net_pnl'].sum()
    survive_pct = t['survives'].mean() * 100
    n_folds     = len(t)

    ticker_summary.append({
        'ticker'      : ticker,
        'n_folds'     : n_folds,
        'mean_ic'     : mean_ic,
        'icir'        : icir,
        'sharpe_gross': mean_sg,
        'sharpe_net'  : mean_sn,
        'net_pnl'     : total_pnl,
        'survive_pct' : survive_pct,
    })

    print(f"{ticker:<8} {n_folds:>6} {mean_ic:>+9.4f} {icir:>7.3f} "
          f"{mean_sg:>+10.3f} {mean_sn:>+10.3f} {total_pnl:>+11,.0f} {survive_pct:>8.1f}%")

print("=" * 95)

# Overall
print("\nOVERALL ACROSS ALL TICKERS AND FOLDS:")
print(f"  Total folds          : {len(res_df)}")
print(f"  Mean IC              : {res_df['ic'].mean():+.4f}")
print(f"  ICIR                 : {res_df['ic'].mean() / res_df['ic'].std():.3f}")
print(f"  Mean Sharpe (gross)  : {res_df['sharpe_gross'].mean():+.3f}")
print(f"  Mean Sharpe (net)    : {res_df['sharpe_net'].mean():+.3f}")
print(f"  Folds surviving      : {res_df['survives'].mean()*100:.1f}%")
print(f"  Total net PnL        : ${res_df['net_pnl'].sum():+,.0f}")

# IC significance test
t_stat, p_val = stats.ttest_1samp(res_df['ic'].dropna(), 0)
print(f"\nIC SIGNIFICANCE (t-test vs zero):")
print(f"  Mean IC  : {res_df['ic'].mean():+.4f}")
print(f"  t-stat   : {t_stat:+.3f}")
print(f"  p-value  : {p_val:.4f}")
print(f"  Significant (p<0.05): {p_val < 0.05}")

# ── DAILY PnL + DSR (new) ─────────────────────────────────────────────────────
daily = pd.concat(DAILY)
daily = daily[(daily['pnl_gross'] != 0) | (daily['pnl_net'] != 0)]
dup = daily.reset_index().duplicated(subset=[daily.index.name or 'index', 'ticker']).sum()
print(f"\nDaily rows: {len(daily)}   duplicate (date,ticker) rows: {dup}  <- must be 0")
daily.to_parquet(os.path.join(PROJECT_ROOT, "features", "wf_daily_pnl.parquet"))

# check: fold results must match the saved walk_forward_results (10m)
saved = pd.read_parquet(os.path.join(PROJECT_ROOT, "features", "walk_forward_results.parquet"))
if 'horizon' in saved.columns:
    saved = saved[saved['horizon'].astype(str).str.contains('10m')]
print(f"Check vs saved file: gross {res_df['sharpe_gross'].mean():+.3f} vs {saved['sharpe_gross'].mean():+.3f}, "
      f"net {res_df['sharpe_net'].mean():+.3f} vs {saved['sharpe_net'].mean():+.3f}  <- must match")

def dsr(r, n_trials):
    r = np.asarray(r, float); n = len(r)
    sr = r.mean() / r.std(ddof=1)
    sk = stats.skew(r); ku = stats.kurtosis(r, fisher=False)
    e = 0.5772156649
    sr0 = 0.0 if n_trials == 1 else np.sqrt(1 / n) * ((1 - e) * stats.norm.ppf(1 - 1 / n_trials)
                                                    + e * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    z = (sr - sr0) * np.sqrt(n - 1) / np.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr ** 2, 1e-12))
    return n, sr * np.sqrt(252), sk, ku - 3, stats.norm.cdf(z)

port = daily.groupby(level=0)[['pnl_gross', 'pnl_net']].sum()
print("\nEQUAL-WEIGHT PORTFOLIO, walk-forward strategy (daily, $1M per ticker)")
print(f"  dates: {port.index.min().date()} -> {port.index.max().date()}")
for col in ('pnl_gross', 'pnl_net'):
    for nt in (1, 25, 300):
        n, sra, sk, exk, d = dsr(port[col], nt)
        print(f"  {col:9s} N_trials={nt:<4d} n={n}  SR_ann={sra:+.3f}  skew={sk:+.2f}  ex_kurt={exk:.2f}  DSR={d:.4f}")
print("\nPer ticker (daily, annualised):")
for tk, g in daily.groupby('ticker'):
    a = g['pnl_gross']; b = g['pnl_net']
    print(f"  {tk:5s} n={len(g):4d}  gross SR={a.mean()/a.std()*np.sqrt(252):+.2f}  net SR={b.mean()/b.std()*np.sqrt(252):+.2f}")
