# gradient_boosting.py
import pandas as pd
import numpy as np
from scipy import stats
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
    'fwd_ret_10m': 60
}

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
    return df

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

def run_fold(train_df, test_df, target, skip):
    feature_cols = [f for f in SELECTED_FEATURES if f in train_df.columns]
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[feature_cols] = train_df[feature_cols].shift(1)
    test_df[feature_cols]  = test_df[feature_cols].shift(1)
    train_clean = train_df[feature_cols + [target]].dropna()
    test_clean  = test_df[feature_cols + [target]].dropna()
    test_nonoverlap = test_clean.iloc[::skip]
    if len(train_clean) < 100 or len(test_nonoverlap) < 10:
        return np.nan, None
    
    X_train = train_clean[feature_cols].values
    y_train = train_clean[target].values
    X_test  = test_nonoverlap[feature_cols].values
    y_test  = test_nonoverlap[target].values

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ic = compute_ic(y_test, y_pred)

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    return ic, importance

# ── RUN ───────────────────────────────────────────────────────────────────────

df = load_data('SPY')
folds = get_wf_folds(df)

print(f"\n{'='*70}")
print(f"LIGHTGBM — ALL HORIZONS — SPY — 13 selected features")
print(f"{'='*70}")
print(f"{'Horizon':<15} {'Mean IC':>8} {'Std IC':>8} {'ICIR':>8} {'T-stat':>8} {'NW T-stat':>10} {'Sig':>5}")
print(f"{'─'*70}")

results = []
all_importances = {}

for target, skip in HORIZONS.items():
    if target not in df.columns:
        continue
    fold_ics = []
    fold_importances = []

    for train_df, test_df in folds:
        ic, importance = run_fold(train_df, test_df, target, skip)
        fold_ics.append(ic)
        if importance:
            fold_importances.append(importance)

    ic_series = pd.Series(fold_ics).dropna()
    mean_ic  = ic_series.mean()
    std_ic   = ic_series.std()
    icir     = mean_ic / std_ic if std_ic > 0 else np.nan
    n        = len(ic_series)
    tstat    = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else np.nan
    tstat_nw = newey_west_tstat(ic_series)
    sig      = 'YES' if abs(tstat_nw) > 2.0 else 'NO'

    print(f"{target:<15} {mean_ic:>8.4f} {std_ic:>8.4f} {icir:>8.4f} {tstat:>8.3f} {tstat_nw:>10.3f} {sig:>5}")

    # Average feature importance across folds
    if fold_importances:
        avg_imp = {}
        for feat in SELECTED_FEATURES:
            avg_imp[feat] = np.mean([d.get(feat, 0) for d in fold_importances])
        top5 = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top features: {[f[0] for f in top5]}")
        all_importances[target] = avg_imp

    results.append({
        'horizon':     target,
        'mean_ic':     round(mean_ic, 4),
        'std_ic':      round(std_ic, 4),
        'icir':        round(icir, 4),
        'tstat':       round(tstat, 3),
        'tstat_nw':    round(tstat_nw, 3),
        'significant': sig
    })

print(f"{'─'*70}")

# Save results
pd.DataFrame(results).to_csv(
    r"E:\quant-research-ofi\reports\SPY_lgbm_all_horizons.csv", index=False)
print(f"\nSaved: SPY_lgbm_all_horizons.csv")

# Final comparison table
print(f"\n{'='*70}")
print(f"MODEL COMPARISON — NW T-STAT — SPY")
print(f"{'='*70}")
print(f"{'Horizon':<15} {'Ridge NW-t':>12} {'Lasso NW-t':>12} {'LightGBM NW-t':>15}")
print(f"{'─'*70}")

ridge_results = {
    'fwd_ret_10s': 1.862,
    'fwd_ret_30s': 3.440,
    'fwd_ret_1m':  2.110,
    'fwd_ret_5m':  0.451,
    'fwd_ret_10m': 0.064
}
lasso_results = {
    'fwd_ret_10s': float('nan'),
    'fwd_ret_30s': -0.057,
    'fwd_ret_1m':  1.092,
    'fwd_ret_5m':  1.165,
    'fwd_ret_10m': 0.202
}

for r in results:
    h = r['horizon']
    ridge_t = ridge_results.get(h, float('nan'))
    lasso_t = lasso_results.get(h, float('nan'))
    lgbm_t  = r['tstat_nw']
    print(f"{h:<15} {ridge_t:>12.3f} {lasso_t:>12.3f} {lgbm_t:>15.3f}")

print(f"{'─'*70}")
print(f"Significant (NW t > 2.0) marked as winner per horizon.")
