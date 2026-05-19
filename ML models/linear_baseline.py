# linear_baseline.py
# Phase 5 ML Models — Ridge Regression Baseline
# Walk-forward IC, ICIR, regular t-stat, Newey-West t-stat
# Includes: lag enforcement, stationary price features, naive persistence baseline
# Reference: Cont, Kukanov & Stoikov (2014) | Lopez de Prado (2018)

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ── FEATURES AND TARGET ───────────────────────────────────────────────────────

# microprice and vwap removed as raw levels — replaced with stationary versions below
FEATURES = [
    'ofi', 'ofi_10s', 'ofi_30s', 'ofi_1m', 'ofi_5m', 'ofi_10m',
    'queue_imbalance', 'trade_imbalance',
    'spread', 'spread_change',
    'microprice_ret',       # log return of microprice — stationary
    'vwap_deviation',       # (microprice - vwap) / spread — stationary
    'kyle_lambda', 'amihud',
    'realized_vol', 'ofi_norm'
]

TARGET = 'fwd_ret_5m'

# ── STATIONARY PRICE FEATURES ─────────────────────────────────────────────────

def add_stationary_price_features(df):
    """
    Replace raw microprice and vwap with stationary versions.
    
    microprice_ret    = log(microprice_t / microprice_{t-1})
                        Why: log return is stationary. Raw level is not.
    
    vwap_deviation    = (microprice - vwap) / spread
                        Why: deviation from VWAP normalised by spread.
                        Tells us: is price above or below the session average,
                        scaled to spread units. Stationary and interpretable.
    
    Manual example:
    microprice = [100.00, 100.02, 99.98]
    microprice_ret = [nan, log(100.02/100.00), log(99.98/100.02)]
                   = [nan, +0.0002, -0.0004]
    
    vwap = 100.01, spread = 0.01
    vwap_deviation = (100.02 - 100.01) / 0.01 = +1.0  (one spread above VWAP)
    """
    df = df.copy()
    
    if 'microprice' in df.columns:
        df['microprice_ret'] = np.log(df['microprice'] / df['microprice'].shift(1))
    else:
        df['microprice_ret'] = np.nan
    
    if 'microprice' in df.columns and 'vwap' in df.columns and 'spread' in df.columns:
        denom = df['spread'].replace(0, np.nan)
        df['vwap_deviation'] = (df['microprice'] - df['vwap']) / denom
    else:
        df['vwap_deviation'] = np.nan
    
    return df

# ── WALK-FORWARD FOLDS ────────────────────────────────────────────────────────

def get_wf_folds(df, train_months=3, test_months=1):
    """
    Build walk-forward folds.
    Each fold: train on 3 months, test on next 1 month, roll forward.
    Returns list of (train_df, test_df) tuples.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    months = df.resample('ME').size().index
    folds = []

    for i in range(train_months, len(months) - test_months + 1):
        train_end = months[i - 1]
        test_end  = months[i + test_months - 1]
        train_df  = df[df.index <= train_end]
        test_df   = df[(df.index > train_end) & (df.index <= test_end)]
        if len(train_df) > 100 and len(test_df) > 100:
            folds.append((train_df, test_df))

    return folds

# ── LAG ENFORCEMENT ───────────────────────────────────────────────────────────

def apply_lag(df):
    """
    Apply shift(1) to ALL features inside the pipeline.
    
    Why: Even if parquet has shift applied, we apply it again here as a
    safety guarantee. shift(1) on already-shifted data = shift(2), which is
    conservative (slightly stale signal) but NEVER leakage.
    
    The alternative — trusting a comment in the parquet — is not acceptable
    when IC validity is at stake.
    
    IMPORTANT: We shift ONLY features, never the target column.
    Target = what actually happened next. Features = what we knew before.
    
    Manual example:
    Row 3: ofi=50, fwd_ret_5m=+0.002
    After shift(1): model sees ofi from row 2 to predict fwd_ret at row 3.
    This is correct. We knew ofi at row 2 before row 3 arrived.
    """
    df = df.copy()
    feature_cols = [f for f in FEATURES if f in df.columns]
    df[feature_cols] = df[feature_cols].shift(1)
    return df

# ── INFORMATION COEFFICIENT ───────────────────────────────────────────────────

def compute_ic(y_true, y_pred):
    """
    IC = Spearman rank correlation between predicted and actual returns.
    Range: -1 to +1. Positive = signal has predictive power.
    Reference: Lopez de Prado (2018) Ch.6
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return np.nan
    ic, _ = stats.spearmanr(y_true[mask], y_pred[mask])
    return ic

# ── NEWEY-WEST T-STAT ─────────────────────────────────────────────────────────

def newey_west_tstat(ic_series, maxlags=5):
    """
    Newey-West corrected t-statistic for IC series.
    
    Why needed: 5-min forward returns overlap with adjacent bars.
    At 10s bar frequency, one 5-min return spans 30 bars.
    Regular t-stat assumes independence across folds → overstates significance.
    HAC (cov_type='HAC') corrects the standard error for autocorrelation.
    
    t_NW = mean_IC / NW_standard_error
    
    Reference: Newey & West (1987) | statsmodels OLS with cov_type='HAC'
    """
    ic_clean = ic_series.dropna()
    if len(ic_clean) < 5:
        return np.nan
    y = ic_clean.values
    X = np.ones((len(y), 1))
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return model.tvalues[0]

# ── NAIVE PERSISTENCE BASELINE ────────────────────────────────────────────────

def naive_persistence_ic(test_df):
    """
    Naive baseline: predict next return = current return.
    
    Why: If Ridge IC barely beats this, the model adds little value.
    The simplest possible benchmark must be beaten before claiming signal.
    
    IC_persistence = spearman(fwd_ret_5m, fwd_ret_5m.shift(1))
    i.e. does last period's return predict this period's return?
    """
    df = test_df[[TARGET]].dropna().copy()
    if len(df) < 10:
        return np.nan
    y_true = df[TARGET].values[1:]
    y_naive = df[TARGET].values[:-1]   # last period's return as prediction
    ic, _ = stats.spearmanr(y_true, y_naive)
    return ic

# ── RIDGE MODEL PER FOLD ──────────────────────────────────────────────────────

def run_fold(train_df, test_df, alpha=1.0):
    # Apply lag inside pipeline — safety guarantee
    train_lagged = apply_lag(train_df)
    test_lagged  = apply_lag(test_df)

    feature_cols = [f for f in FEATURES if f in train_lagged.columns]

    train_clean = train_lagged[feature_cols + [TARGET]].dropna()
    test_clean  = test_lagged[feature_cols + [TARGET]].dropna()

    if len(train_clean) < 100 or len(test_clean) < 10:
        return np.nan, np.nan, None, None, None

    X_train = train_clean[feature_cols].values
    y_train = train_clean[TARGET].values

    # Non-overlapping test rows only — every 30th row
    # fwd_ret_5m spans 30 bars at 10s frequency
    # Adjacent rows share 29/30 bars — naive IC = 0.96 from overlap not signal
    test_nonoverlap = test_clean.iloc[::30]
    X_test  = test_nonoverlap[feature_cols].values
    y_test  = test_nonoverlap[TARGET].values

    # Scaler: fit on train ONLY
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    cond_number = np.linalg.cond(X_train_sc)

    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_sc, y_train)
    y_pred = ridge.predict(X_test_sc)

    ic_ridge = compute_ic(y_test, y_pred)
    ic_naive = naive_persistence_ic(test_nonoverlap)

    return ic_ridge, ic_naive, y_test, y_pred, cond_number
# ── MAIN: RUN WALK-FORWARD ────────────────────────────────────────────────────

def run_linear_baseline(ticker='SPY', alpha=1.0):
    """
    Full walk-forward Ridge baseline for one ticker.
    Prints fold-by-fold IC table, summary stats, NW t-stat.
    Saves CSV report to reports folder.
    """
    path = rf"E:\quant-research-ofi\features\{ticker}_features.parquet"
    df = pd.read_parquet(path)
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index('ts').sort_index()

    # Add stationary price features
    df = add_stationary_price_features(df)

    print(f"\n{'='*60}")
    print(f"LINEAR BASELINE — {ticker}")
    print(f"Rows: {len(df):,} | Target: {TARGET}")
    print(f"{'='*60}")

    folds = get_wf_folds(df, train_months=3, test_months=1)
    print(f"Walk-forward folds: {len(folds)}\n")

    fold_ics      = []
    fold_ics_naive = []
    fold_details  = []

    for i, (train_df, test_df) in enumerate(folds):
        ic, ic_naive, y_true, y_pred, cond = run_fold(train_df, test_df, alpha=alpha)

        fold_ics.append(ic)
        fold_ics_naive.append(ic_naive)

        fold_details.append({
            'fold':        i + 1,
            'train_start': train_df.index[0].date(),
            'train_end':   train_df.index[-1].date(),
            'test_start':  test_df.index[0].date(),
            'test_end':    test_df.index[-1].date(),
            'test_rows':   len(test_df),
            'ic_ridge':    round(ic, 4) if not np.isnan(ic) else np.nan,
            'ic_naive':    round(ic_naive, 4) if not np.isnan(ic_naive) else np.nan,
            'ridge_beats_naive': ic > ic_naive if not (np.isnan(ic) or np.isnan(ic_naive)) else np.nan,
            'cond_number': round(cond, 1) if cond is not None else np.nan
        })

        print(f"  Fold {i+1}: {train_df.index[0].date()}→{train_df.index[-1].date()} "
              f"| test {test_df.index[0].date()}→{test_df.index[-1].date()} "
              f"| IC_ridge={ic:.4f} | IC_naive={ic_naive:.4f} "
              f"| beats_naive={'YES' if ic > ic_naive else 'NO'}")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    ic_series    = pd.Series(fold_ics).dropna()
    naive_series = pd.Series(fold_ics_naive).dropna()

    mean_ic  = ic_series.mean()
    std_ic   = ic_series.std()
    icir     = mean_ic / std_ic if std_ic > 0 else np.nan
    n        = len(ic_series)

    tstat_regular = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else np.nan
    pval_regular  = 2 * (1 - stats.t.cdf(abs(tstat_regular), df=n - 1))
    tstat_nw      = newey_west_tstat(ic_series, maxlags=5)

    print(f"\n{'─'*60}")
    print(f"RIDGE BASELINE RESULTS — {ticker}")
    print(f"{'─'*60}")
    print(f"  Folds            : {n}")
    print(f"  Mean IC (Ridge)  : {mean_ic:.4f}")
    print(f"  Mean IC (Naive)  : {naive_series.mean():.4f}")
    print(f"  Std IC           : {std_ic:.4f}")
    print(f"  ICIR             : {icir:.4f}")
    print(f"  T-stat (regular) : {tstat_regular:.3f}  |  p-value: {pval_regular:.4f}")
    print(f"  T-stat (NW HAC)  : {tstat_nw:.3f}")
    print(f"  Significant (NW) : {'YES — signal survives autocorrelation correction' if abs(tstat_nw) > 2.0 else 'NO — inflated by autocorrelation'}")
    print(f"{'─'*60}")

    fold_df = pd.DataFrame(fold_details)
    print(f"\nFold-by-fold table:")
    print(fold_df.to_string(index=False))

    out_path = rf"E:\quant-research-ofi\reports\{ticker}_ridge_baseline.csv"
    fold_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    summary = {
        'ticker':         ticker,
        'n_folds':        n,
        'mean_ic':        round(mean_ic, 4),
        'mean_ic_naive':  round(naive_series.mean(), 4),
        'std_ic':         round(std_ic, 4),
        'icir':           round(icir, 4),
        'tstat_regular':  round(tstat_regular, 3),
        'pval_regular':   round(pval_regular, 4),
        'tstat_nw':       round(tstat_nw, 3),
        'significant_nw': abs(tstat_nw) > 2.0
    }

    return summary


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    summary = run_linear_baseline(ticker='SPY', alpha=1.0)

    print(f"\n{'='*60}")
    print("BENCHMARK SET. All future models compared to this.")
    print(f"Mean IC: {summary['mean_ic']} | ICIR: {summary['icir']} | NW t: {summary['tstat_nw']}")
    print(f"{'='*60}")
