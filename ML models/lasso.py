# lasso_all_horizons.py
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
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

def run_fold(train_df, test_df, target, skip, alpha=0.000001):
    feature_cols = [f for f in SELECTED_FEATURES if f in train_df.columns]
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[feature_cols] = train_df[feature_cols].shift(1)
    test_df[feature_cols]  = test_df[feature_cols].shift(1)
    train_clean = train_df[feature_cols + [target]].dropna()
    test_clean  = test_df[feature_cols + [target]].dropna()
    test_nonoverlap = test_clean.iloc[::skip]
    if len(train_clean) < 100 or len(test_nonoverlap) < 10:
        return np.nan, []
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_clean[feature_cols].values)
    X_test  = scaler.transform(test_nonoverlap[feature_cols].values)
    y_train = train_clean[target].values
    y_test  = test_nonoverlap[target].values
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    # Track which features survived
    nonzero = [feature_cols[i] for i, c in enumerate(lasso.coef_) if c != 0]
    return compute_ic(y_test, y_pred), nonzero

# ── RUN ───────────────────────────────────────────────────────────────────────

df = load_data('SPY')
folds = get_wf_folds(df)

print(f"\n{'='*65}")
print(f"LASSO BASELINE — ALL HORIZONS — SPY — 13 selected features")
print(f"{'='*65}")
print(f"{'Horizon':<15} {'Mean IC':>8} {'Std IC':>8} {'ICIR':>8} {'T-stat':>8} {'NW T-stat':>10} {'Sig':>5}")
print(f"{'─'*65}")

results = []

for target, skip in HORIZONS.items():
    if target not in df.columns:
        continue
    fold_ics = []
    all_nonzero = []

    for train_df, test_df in folds:
        ic, nonzero = run_fold(train_df, test_df, target, skip)
        fold_ics.append(ic)
        all_nonzero.extend(nonzero)

    ic_series = pd.Series(fold_ics).dropna()
    mean_ic  = ic_series.mean()
    std_ic   = ic_series.std()
    icir     = mean_ic / std_ic if std_ic > 0 else np.nan
    n        = len(ic_series)
    tstat    = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else np.nan
    tstat_nw = newey_west_tstat(ic_series)
    sig      = 'YES' if abs(tstat_nw) > 2.0 else 'NO'

    # Most selected features across folds
    from collections import Counter
    top_features = Counter(all_nonzero).most_common(5)

    print(f"{target:<15} {mean_ic:>8.4f} {std_ic:>8.4f} {icir:>8.4f} {tstat:>8.3f} {tstat_nw:>10.3f} {sig:>5}")
    print(f"  Top features selected by Lasso: {[f[0] for f in top_features]}")

    results.append({
        'horizon':     target,
        'mean_ic':     round(mean_ic, 4),
        'std_ic':      round(std_ic, 4),
        'icir':        round(icir, 4),
        'tstat':       round(tstat, 3),
        'tstat_nw':    round(tstat_nw, 3),
        'significant': sig,
        'top_features': str([f[0] for f in top_features])
    })

print(f"{'─'*65}")
pd.DataFrame(results).to_csv(
    r"E:\quant-research-ofi\reports\SPY_lasso_all_horizons.csv", index=False)
print(f"\nSaved: SPY_lasso_all_horizons.csv")
