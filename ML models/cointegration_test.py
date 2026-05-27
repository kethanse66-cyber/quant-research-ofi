# cointegration_test.py
# Phase 5 ML Models — Cointegration Analysis
# True rolling hedge ratio using RollingOLS — no lookahead bias
# OOS spread IC with block bootstrap CI
# Engle-Granger + Johansen tests
# Reference: Engle & Granger (1987) | Johansen (1991)

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# ── LOAD DATA ─────────────────────────────────────────────────────

def load_ticker(ticker):
    path = rf"E:\quant-research-ofi\features\{ticker}_features.parquet"
    df = pd.read_parquet(path)
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index('ts').sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def get_log_midprice(df):
    """
    Log midprice resampled to 1-minute bars.
    Why log: additive, reduces scale differences across assets.
    Why 1-min: raw tick microprice too noisy for cointegration.
    Cointegration is a low-frequency relationship.
    """
    if 'microprice' in df.columns:
        price = df['microprice']
    elif 'weighted_mid' in df.columns:
        price = df['weighted_mid']
    else:
        raise ValueError(f"No price column found in {df.columns.tolist()}")

    price_1m = price.resample('1min').last().dropna()
    return np.log(price_1m)

# ── ADF TEST ──────────────────────────────────────────────────────

def adf_test(series, name):
    result = adfuller(series.dropna(), autolag='AIC')
    pval = result[1]
    stationary = pval < 0.05
    print(f"  ADF — {name}: p={pval:.4f} — {'STATIONARY' if stationary else 'NON-STATIONARY'}")
    return stationary

# ── ENGLE-GRANGER TEST ────────────────────────────────────────────

def engle_granger_test(s1, s2, name1, name2):
    combined = pd.DataFrame({'s1': s1, 's2': s2}).dropna()
    t_stat, p_value, _ = coint(combined['s1'], combined['s2'])
    cointegrated = p_value < 0.05
    print(f"\n  Engle-Granger {name1} vs {name2}: t={t_stat:.3f} p={p_value:.4f} — {'COINTEGRATED' if cointegrated else 'NOT COINTEGRATED'}")
    return cointegrated, p_value

# ── JOHANSEN TEST ─────────────────────────────────────────────────

def johansen_test(series_dict):
    """
    Johansen test — handles multiple series simultaneously.
    More powerful than Engle-Granger for multi-asset systems.
    Tests number of cointegrating vectors among N series.
    """
    names = list(series_dict.keys())
    combined = pd.DataFrame(series_dict).dropna()
    print(f"\n  Johansen test — {' + '.join(names)} | obs={len(combined)}")
    result = coint_johansen(combined.values, det_order=0, k_ar_diff=1)
    print(f"  Trace statistics vs 95% critical values:")
    for i in range(len(names)):
        trace = result.lr1[i]
        crit  = result.cvt[i, 1]
        reject = trace > crit
        print(f"  r<={i}: trace={trace:.3f} crit={crit:.3f} — {'REJECT null' if reject else 'FAIL TO REJECT'}")
    return result

# ── ROLLING HEDGE RATIO — FAST VERSION ───────────────────────────

def rolling_hedge_ratio(s1, s2, window_days=60):
    """
    Rolling hedge ratio using statsmodels RollingOLS.
    Much faster than loop-based OLS.

    TRUE rolling window: uses only last window_days of data at each point.
    Why rolling not expanding: beta must adapt to regime changes.
    Expanding window gives too much weight to ancient data.

    At each point T: beta estimated on [T-window, T-1] only.
    Spread at T = s1_T - beta_{T-1} * s2_T
    This guarantees no lookahead bias.

    Economic intuition for SPY-ES1! spread:
    SPY and ES1! track same index. Arbitrageurs keep them aligned.
    Spread = deviation from fair value. Predictive during arbitrage latency window.
    When SPY rich vs ES1! → SPY likely to fall or ES1! to rise.
    This is cross-market price discovery — seconds to minutes horizon.
    """
    combined = pd.DataFrame({'s1': s1, 's2': s2}).dropna()

    # Convert window to bars (1-min bars: ~390 per day)
    window_bars = window_days * 390

    print(f"\n  Rolling hedge ratio — window={window_days} days ({window_bars} bars)")
    print(f"  Using statsmodels RollingOLS — fast vectorised computation")

    # Add constant for intercept
    endog = combined['s1']
    exog  = sm.add_constant(combined['s2'])

    # RollingOLS — true rolling window, not expanding
    rolling_model = RollingOLS(endog, exog, window=window_bars)
    rolling_result = rolling_model.fit()

    betas  = rolling_result.params['s2']
    alphas = rolling_result.params['const']

    # Shift by 1 — use yesterday's beta to compute today's spread
    # This is the lookahead prevention — same principle as shift(1) for features
    betas_lagged  = betas.shift(1)
    alphas_lagged = alphas.shift(1)

    spread = combined['s1'] - betas_lagged * combined['s2'] - alphas_lagged
    spread = spread.dropna()

    print(f"  Valid spread observations: {len(spread)}")
    print(f"  Spread mean: {spread.mean():.6f} | std: {spread.std():.6f}")

    return spread, betas_lagged.dropna()

# ── BLOCK BOOTSTRAP IC CI ─────────────────────────────────────────

def block_bootstrap_ic(x, y, n_boot=500, block_size=30):
    """
    Block bootstrap confidence interval for IC.
    
    Why block bootstrap not IID bootstrap:
    Time series observations are autocorrelated.
    IID bootstrap breaks the temporal structure — wrong for finance.
    Block bootstrap samples contiguous blocks — preserves autocorrelation.
    Block size = 30 matches our overlap correction (5-min = 30 bars at 10s).
    
    Reference: Kunsch (1989) | cited in blueprint Section 3.1
    """
    n = len(x)
    boot_ics = []

    for _ in range(n_boot):
        # Sample block start indices
        n_blocks = int(np.ceil(n / block_size))
        block_starts = np.random.randint(0, n - block_size, n_blocks)
        indices = []
        for start in block_starts:
            indices.extend(range(start, start + block_size))
        indices = np.array(indices[:n])
        boot_ic, _ = scipy_stats.spearmanr(x[indices], y[indices])
        boot_ics.append(boot_ic)

    ci_low  = np.percentile(boot_ics, 2.5)
    ci_high = np.percentile(boot_ics, 97.5)
    return ci_low, ci_high

# ── OOS IC CHECK ──────────────────────────────────────────────────

def oos_ic_check(spread, target_df, train_frac=0.75,
                 target_col='fwd_ret_30s', skip=3):
    """
    Out-of-sample IC check with block bootstrap CI.

    Split: first train_frac of data for hedge ratio estimation.
    Test: remaining data — never used in estimation.

    Frequency alignment:
    Spread is at 1-min frequency.
    Targets are at 10s bar frequency.
    We forward-fill spread to 10s frequency — each 10s bar gets
    the most recent 1-min spread value.
    Shift by 1 bar applied AFTER ffill to prevent lookahead.
    This guarantees: spread used at time T was known before T.
    """
    # Merge spread with target
    df = pd.DataFrame({
        'spread': spread,
        'target': target_df[target_col]
    }).dropna()

    # OOS split
    split_idx = int(len(df) * train_frac)
    train_end = df.index[split_idx]
    test = df[df.index > train_end].copy()

    # Lag spread — no lookahead
    test['spread_lagged'] = test['spread'].shift(1)
    test = test.dropna().iloc[::skip]

    if len(test) < 50:
        print(f"  OOS IC: insufficient test data ({len(test)} rows)")
        return np.nan

    ic, pval = scipy_stats.spearmanr(
        test['spread_lagged'], test['target'])

    # Block bootstrap CI
    ci_low, ci_high = block_bootstrap_ic(
        test['spread_lagged'].values,
        test['target'].values,
        block_size=30
    )

    print(f"\n  OOS IC vs {target_col}: {ic:.4f} | p={pval:.4f}")
    print(f"  Block bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Test observations: {len(test)}")
    print(f"  Train end: {train_end.date()}")
    sig = 'SIGNIFICANT' if pval < 0.05 and abs(ic) > 0.005 else 'NOT SIGNIFICANT'
    print(f"  Result: {sig}")

    return ic

# ── MAIN ──────────────────────────────────────────────────────────

def run_cointegration_analysis():
    print(f"\n{'='*65}")
    print(f"COINTEGRATION ANALYSIS")
    print(f"{'='*65}")

    # Load tickers
    spy = load_ticker('SPY')
    qqq = load_ticker('QQQ')
    iwm = load_ticker('IWM')

    print("\nComputing log midprices at 1-min frequency...")
    spy_log = get_log_midprice(spy)
    qqq_log = get_log_midprice(qqq)
    iwm_log = get_log_midprice(iwm)

    print(f"SPY: {len(spy_log)} bars | QQQ: {len(qqq_log)} bars | IWM: {len(iwm_log)} bars")

    try:
        es1 = load_ticker('ES1!')
        es1_log = get_log_midprice(es1)
        print(f"ES1!: {len(es1_log)} bars")
        has_es1 = True
    except:
        print("ES1! not available")
        has_es1 = False

    # ── STEP 1: STATIONARITY ──────────────────────────────────────
    print(f"\n{'─'*65}")
    print("STEP 1 — Stationarity (series must be non-stationary for cointegration)")
    print(f"{'─'*65}")
    adf_test(spy_log, 'SPY log-price')
    adf_test(qqq_log, 'QQQ log-price')
    adf_test(iwm_log, 'IWM log-price')
    if has_es1:
        adf_test(es1_log, 'ES1! log-price')

    # ── STEP 2: ENGLE-GRANGER ─────────────────────────────────────
    print(f"\n{'─'*65}")
    print("STEP 2 — Engle-Granger pairwise cointegration")
    print(f"{'─'*65}")

    pairs = [('SPY', spy_log, 'QQQ', qqq_log),
             ('SPY', spy_log, 'IWM', iwm_log),
             ('QQQ', qqq_log, 'IWM', iwm_log)]
    if has_es1:
        pairs.insert(0, ('SPY', spy_log, 'ES1!', es1_log))

    coint_results = {}
    for n1, s1, n2, s2 in pairs:
        c, p = engle_granger_test(s1, s2, n1, n2)
        coint_results[f'{n1}_{n2}'] = c

    # ── STEP 3: JOHANSEN ──────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("STEP 3 — Johansen multi-asset test")
    print(f"{'─'*65}")
    multi = pd.DataFrame({
        'SPY': spy_log, 'QQQ': qqq_log, 'IWM': iwm_log
    }).dropna()
    johansen_test(multi.to_dict('series'))

    # ── STEP 4: ROLLING SPREAD + OOS IC ───────────────────────────
    print(f"\n{'─'*65}")
    print("STEP 4 — Rolling hedge ratio and OOS IC")
    print(f"{'─'*65}")

    results = []

    for n1, s1, n2, s2 in pairs:
        print(f"\n  {'─'*50}")
        print(f"  Pair: {n1} vs {n2}")

        spread, betas = rolling_hedge_ratio(s1, s2, window_days=60)

        if len(spread) < 100:
            print(f"  Insufficient spread data — skip")
            continue

        # Spread stationarity check
        spread_stat = adf_test(spread, f'{n1}-{n2} spread')

        # Resample spread to 10s bars for IC check
        # Forward fill — each 10s bar gets most recent 1-min spread
        spread_10s = spread.reindex(spy.index, method='ffill')

        # OOS IC
        ic_30s = oos_ic_check(spread_10s, spy, 0.75, 'fwd_ret_30s', 3)
        ic_1m  = oos_ic_check(spread_10s, spy, 0.75, 'fwd_ret_1m',  6)

        results.append({
            'pair':              f'{n1}_{n2}',
            'cointegrated_EG':   coint_results.get(f'{n1}_{n2}', False),
            'spread_stationary': spread_stat,
            'oos_ic_30s':        round(ic_30s, 4) if not np.isnan(ic_30s) else np.nan,
            'oos_ic_1m':         round(ic_1m,  4) if not np.isnan(ic_1m)  else np.nan,
        })

        # Save spread if worth adding
        if spread_stat and (abs(ic_30s) > 0.005 or abs(ic_1m) > 0.005):
            out = rf"E:\quant-research-ofi\features\{n1}_{n2}_spread.parquet"
            pd.DataFrame({f'{n1}_{n2}_spread': spread_10s}).to_parquet(out)
            print(f"\n  Spread saved — add {n1}_{n2}_spread to SELECTED_FEATURES")
        else:
            print(f"\n  Spread IC too weak — not adding to features")

    # ── SUMMARY ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("COINTEGRATION SUMMARY")
    print(f"{'='*65}")
    if results:
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        df_results.to_csv(
            r"E:\quant-research-ofi\reports\cointegration_results.csv",
            index=False)
        print(f"\nSaved: cointegration_results.csv")


if __name__ == "__main__":
    run_cointegration_analysis()
