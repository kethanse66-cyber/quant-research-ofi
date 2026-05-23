# ic_summary_table.py
# Phase 5 ML Models — Final IC Summary Table
# All models, all horizons, NW t-stats side by side
# This table goes directly into paper Section 6
# Reference: Lopez de Prado (2018) | Newey & West (1987)

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

SELECTED_FEATURES = [
    'ofi_10s', 'ofi_30s', 'ofi_10m',
    'queue_imbalance', 'trade_imbalance',
    'spread', 'spread_change',
    'kyle_lambda', 'amihud',
    'realized_vol', 'ofi_norm',
    'microprice_ret', 'vwap_deviation'
]

HORIZONS = {
    'fwd_ret_10s': 1,
    'fwd_ret_30s': 3,
    'fwd_ret_1m':  6,
    'fwd_ret_5m':  30,
    'fwd_ret_10m': 60,
}

# ── DATA LOADING ──────────────────────────────────────────────────

def load_data(ticker='SPY'):
    path = rf"E:\quant-research-ofi\features\{ticker}_features.parquet"
    df = pd.read_parquet(path)
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index('ts').sort_index()
    if 'microprice' in df.columns:
        df['microprice_ret'] = np.log(df['microprice'] / df['microprice'].shift(1))
    if 'microprice' in df.columns and 'vwap' in df.columns and 'spread' in df.columns:
        denom = df['spread'].replace(0, np.nan)
        df['vwap_deviation'] = (df['microprice'] - df['vwap']) / denom

    # Merge regime labels
    regime_path = rf"E:\quant-research-ofi\features\{ticker}_features_with_regimes.parquet"
    regimes = pd.read_parquet(regime_path)[['regime']]
    regimes.index = pd.to_datetime(regimes.index).tz_localize(None)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.join(regimes, how='left')
    return df

def get_wf_folds(df, train_months=3, test_months=1):
    df = df.copy()
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

# ── CORE FUNCTIONS ────────────────────────────────────────────────

def compute_ic(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return np.nan
    ic, _ = stats.spearmanr(y_true[mask], y_pred[mask])
    return ic

def newey_west_tstat(ic_series, maxlags=10):
    ic_clean = ic_series.dropna()
    if len(ic_clean) < 5:
        return np.nan
    y = ic_clean.values
    X = np.ones((len(y), 1))
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return model.tvalues[0]

def summarise(fold_ics):
    s = pd.Series(fold_ics).dropna()
    if len(s) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean_ic  = s.mean()
    std_ic   = s.std()
    icir     = mean_ic / std_ic if std_ic > 0 else np.nan
    tstat_nw = newey_west_tstat(s)
    return mean_ic, std_ic, icir, tstat_nw

# ── MODEL RUNNERS ─────────────────────────────────────────────────

def run_ridge(train_df, test_df, target, skip, regime=None):
    """
    Ridge regression. If regime specified, train only on that regime.
    regime=None → global model
    regime=0    → calm only
    regime=2    → stressed only
    """
    feature_cols = [f for f in SELECTED_FEATURES if f in train_df.columns]
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[feature_cols] = train_df[feature_cols].shift(1)
    test_df[feature_cols]  = test_df[feature_cols].shift(1)

    cols = feature_cols + [target] + (['regime'] if regime is not None else [])
    train_clean = train_df[cols].dropna()
    test_clean  = test_df[cols].dropna()

    # Filter train by regime if specified
    if regime is not None:
        train_clean = train_clean[train_clean['regime'] == regime]
        test_clean  = test_clean[test_clean['regime'] == regime]

    test_nonoverlap = test_clean.iloc[::skip]

    if len(train_clean) < 50 or len(test_nonoverlap) < 5:
        return np.nan

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_clean[feature_cols].values)
    X_test  = scaler.transform(test_nonoverlap[feature_cols].values)
    y_train = train_clean[target].values
    y_test  = test_nonoverlap[target].values

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return compute_ic(y_test, model.predict(X_test))

def run_lasso(train_df, test_df, target, skip):
    feature_cols = [f for f in SELECTED_FEATURES if f in train_df.columns]
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[feature_cols] = train_df[feature_cols].shift(1)
    test_df[feature_cols]  = test_df[feature_cols].shift(1)
    train_clean = train_df[feature_cols + [target]].dropna()
    test_clean  = test_df[feature_cols + [target]].dropna()
    test_nonoverlap = test_clean.iloc[::skip]
    if len(train_clean) < 50 or len(test_nonoverlap) < 5:
        return np.nan
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_clean[feature_cols].values)
    X_test  = scaler.transform(test_nonoverlap[feature_cols].values)
    model = Lasso(alpha=0.000001, max_iter=10000)
    model.fit(X_train, train_clean[target].values)
    return compute_ic(test_nonoverlap[target].values, model.predict(X_test))

def run_lgbm(train_df, test_df, target, skip):
    feature_cols = [f for f in SELECTED_FEATURES if f in train_df.columns]
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[feature_cols] = train_df[feature_cols].shift(1)
    test_df[feature_cols]  = test_df[feature_cols].shift(1)
    train_clean = train_df[feature_cols + [target]].dropna()
    test_clean  = test_df[feature_cols + [target]].dropna()
    test_nonoverlap = test_clean.iloc[::skip]
    if len(train_clean) < 50 or len(test_nonoverlap) < 5:
        return np.nan
    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, num_leaves=15,
        min_child_samples=50, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbose=-1
    )
    model.fit(train_clean[feature_cols].values, train_clean[target].values)
    return compute_ic(test_nonoverlap[target].values,
                      model.predict(test_nonoverlap[feature_cols].values))

# ── MAIN ──────────────────────────────────────────────────────────

def build_ic_summary(ticker='SPY'):
    print(f"\n{'='*75}")
    print(f"IC SUMMARY TABLE — {ticker} — ALL MODELS — ALL HORIZONS")
    print(f"{'='*75}")

    df = load_data(ticker)
    folds = get_wf_folds(df)
    print(f"Folds: {len(folds)}\n")

    # Models to run
    models = {
        'Ridge-Global':   lambda tr, te, tgt, sk: run_ridge(tr, te, tgt, sk, regime=None),
        'Ridge-Calm':     lambda tr, te, tgt, sk: run_ridge(tr, te, tgt, sk, regime=0),
        'Ridge-Stressed': lambda tr, te, tgt, sk: run_ridge(tr, te, tgt, sk, regime=2),
        'Lasso':          run_lasso,
        'LightGBM':       run_lgbm,
    }

    all_rows = []

    for target, skip in HORIZONS.items():
        if target not in df.columns:
            continue

        print(f"Running horizon: {target}")

        for model_name, model_fn in models.items():
            fold_ics = []
            for train_df, test_df in folds:
                ic = model_fn(train_df, test_df, target, skip)
                fold_ics.append(ic)

            mean_ic, std_ic, icir, tstat_nw = summarise(fold_ics)
            sig = 'YES' if not np.isnan(tstat_nw) and abs(tstat_nw) > 2.0 else 'NO'

            all_rows.append({
                'horizon':   target,
                'model':     model_name,
                'mean_ic':   round(mean_ic, 4) if not np.isnan(mean_ic) else np.nan,
                'std_ic':    round(std_ic, 4)  if not np.isnan(std_ic)  else np.nan,
                'icir':      round(icir, 4)    if not np.isnan(icir)    else np.nan,
                'tstat_nw':  round(tstat_nw, 3) if not np.isnan(tstat_nw) else np.nan,
                'significant': sig,
            })

    # ── PRINT TABLE ───────────────────────────────────────────────
    summary = pd.DataFrame(all_rows)

    print(f"\n{'─'*75}")
    print(f"{'Horizon':<15} {'Model':<18} {'Mean IC':>8} {'Std IC':>8} {'ICIR':>7} {'NW-t':>8} {'Sig':>5}")
    print(f"{'─'*75}")

    for horizon in HORIZONS.keys():
        sub = summary[summary['horizon'] == horizon]
        for _, row in sub.iterrows():
            sig_marker = '***' if row['significant'] == 'YES' else '   '
            print(f"{row['horizon']:<15} {row['model']:<18} "
                  f"{row['mean_ic']:>8.4f} {row['std_ic']:>8.4f} "
                  f"{row['icir']:>7.4f} {row['tstat_nw']:>8.3f} {sig_marker}")
        print(f"{'─'*75}")

    # ── SAVE ──────────────────────────────────────────────────────
    out_path = rf"E:\quant-research-ofi\reports\{ticker}_ic_summary_table.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # ── BEST RESULTS ──────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("BEST SIGNIFICANT RESULTS — ranked by NW t-stat")
    print(f"{'='*75}")
    sig_only = summary[summary['significant'] == 'YES'].copy()
    sig_only = sig_only.sort_values('tstat_nw', ascending=False)
    print(sig_only[['horizon','model','mean_ic','icir','tstat_nw']].to_string(index=False))

    return summary


if __name__ == "__main__":
    summary = build_ic_summary(ticker='SPY')
