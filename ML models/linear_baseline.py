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

FEATURES = [
    'ofi', 'ofi_10s', 'ofi_30s', 'ofi_1m', 'ofi_5m', 'ofi_10m',
    'queue_imbalance', 'trade_imbalance',
    'spread', 'spread_change',
    'microprice_ret',
    'vwap_deviation',
    'kyle_lambda', 'amihud',
    'realized_vol', 'ofi_norm'
]

TARGET = 'fwd_ret_5m'

# ── STATIONARY PRICE FEATURES ─────────────────────────────────────────────────

def add_stationary_price_features(df):
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
    df = df.copy()
    feature_cols = [f for f in FEATURES if f in df.columns]
    df[feature_cols] = df[feature_cols].shift(1)
    return df

# ── INFORMATION COEFFICIENT ───────────────────────────────────────────────────

def compute_ic(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return np.nan
    ic, _ = stats.spearmanr(y_true[mask], y_pred[mask])
    return ic

# ── NEWEY-WEST T-STAT ─────────────────────────────────────────────────────────

def newey_west_tstat(ic_series, maxlags=5):
    ic_clean = ic_series.dropna()
    if len(ic_clean) < 5:
        return np.nan
    y = ic_clean.values
    X = np.ones((len(y), 1))
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return model.tvalues[0]

# ── NAIVE PERSISTENCE BASELINE ────────────────────────────────────────────────

def naive_persistence_ic(test_df):
    df = test_df[[TARGET]].dropna().copy()
    if len(df) < 10:
        return np.nan
    y_true  = df[TARGET].values[1:]
    y_naive = df[TARGET].values[:-1]
    ic, _ = stats.spearmanr(y_true, y_naive)
    return ic

# ── RIDGE MODEL PER FOLD ──────────────────────────────────────────────────────

def run_fold(train_df, test_df, alpha=1.0):
    train_lagged = apply_lag(train_df)
    test_lagged  = apply_lag(test_df)

    feature_cols = [f for f in FEATURES if f in train_lagged.columns]

    train_clean = train_lagged[feature_cols + [TARGET]].dropna()
    test_clean  = test_lagged[feature_cols + [TARGET]].dropna()

    if len(train_clean) < 100 or len(test_clean) < 10:
        return np.nan, np.nan, None, None, None, None

    X_train = train_clean[feature_cols].values
    y_train = train_clean[TARGET].values

    # Non-overlapping test rows only — every 30th row
    test_nonoverlap = test_clean.iloc[::30]
    X_test  = test_nonoverlap[feature_cols].values
    y_test  = test_nonoverlap[TARGET].values

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    cond_number = np.linalg.cond(X_train_sc)

    ridge  = Ridge(alpha=alpha)
    ridge.fit(X_train_sc, y_train)
    y_pred = ridge.predict(X_test_sc)

    ic_ridge = compute_ic(y_test, y_pred)
    ic_naive = naive_persistence_ic(test_nonoverlap)

    # Save predictions with timestamps for DSR calculator
    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    }, index=test_nonoverlap.index)

    return ic_ridge, ic_naive, y_test, y_pred, cond_number, pred_df

# ── MAIN: RUN WALK-FORWARD ────────────────────────────────────────────────────

def run_linear_baseline(ticker='SPY', alpha=1.0):
    path = rf"E:\quant-research-ofi\features\{ticker}_features.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = add_stationary_price_features(df)

    print(f"\n{'='*60}")
    print(f"LINEAR BASELINE — {ticker}")
    print(f"Rows: {len(df):,} | Target: {TARGET}")
    print(f"{'='*60}")

    folds = get_wf_folds(df, train_months=3, test_months=1)
    print(f"Walk-forward folds: {len(folds)}\n")

    fold_ics       = []
    fold_ics_naive = []
    fold_details   = []
    all_preds      = []   # collect bar-level predictions across all folds

    for i, (train_df, test_df) in enumerate(folds):
        ic, ic_naive, y_true, y_pred, cond, pred_df = run_fold(train_df, test_df, alpha=alpha)

        fold_ics.append(ic)
        fold_ics_naive.append(ic_naive)

        # Collect predictions if fold ran successfully
        if pred_df is not None:
            all_preds.append(pred_df)

        fold_details.append({
            'fold':              i + 1,
            'train_start':       train_df.index[0].date(),
            'train_end':         train_df.index[-1].date(),
            'test_start':        test_df.index[0].date(),
            'test_end':          test_df.index[-1].date(),
            'test_rows':         len(test_df),
            'ic_ridge':          round(ic, 4) if not np.isnan(ic) else np.nan,
            'ic_naive':          round(ic_naive, 4) if not np.isnan(ic_naive) else np.nan,
            'ridge_beats_naive': ic > ic_naive if not (np.isnan(ic) or np.isnan(ic_naive)) else np.nan,
            'cond_number':       round(cond, 1) if cond is not None else np.nan
        })

        print(f"  Fold {i+1}: {train_df.index[0].date()}→{train_df.index[-1].date()} "
              f"| test {test_df.index[0].date()}→{test_df.index[-1].date()} "
              f"| IC_ridge={ic:.4f} | IC_naive={ic_naive:.4f} "
              f"| beats_naive={'YES' if ic > ic_naive else 'NO'}")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    ic_series    = pd.Series(fold_ics).dropna()
    naive_series = pd.Series(fold_ics_naive).dropna()

    mean_ic = ic_series.mean()
    std_ic  = ic_series.std()
    icir    = mean_ic / std_ic if std_ic > 0 else np.nan
    n       = len(ic_series)

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
    print(f"  Significant (NW) : {'YES' if abs(tstat_nw) > 2.0 else 'NO'}")
    print(f"{'─'*60}")

    fold_df = pd.DataFrame(fold_details)
    print(f"\nFold-by-fold table:")
    print(fold_df.to_string(index=False))

    out_path = rf"E:\quant-research-ofi\reports\{ticker}_ridge_baseline.csv"
    fold_df.to_csv(out_path, index=False)
    print(f"\nSaved fold summary: {out_path}")

    # Save bar-level predictions for DSR calculator
    if all_preds:
        preds_combined = pd.concat(all_preds).sort_index()
        preds_path     = rf"E:\quant-research-ofi\features\{ticker}_ypred_ytrue.parquet"
        preds_combined.to_parquet(preds_path)
        print(f"Saved bar-level predictions: {preds_path}")
        print(f"Shape: {preds_combined.shape} | Columns: {preds_combined.columns.tolist()}")

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
