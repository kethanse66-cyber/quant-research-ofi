# per_regime_models.py
# Phase 5 ML Models — Per-Regime Ridge Models
# Train separate Ridge per regime, compare IC vs global model
# Matched evaluation: train calm → test calm, train stressed → test stressed
# Reference: Cont, Kukanov & Stoikov (2014) | Lopez de Prado (2018)

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import os
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

def load_data(ticker='SPY'):
    # Load full features
    path = rf"E:\quant-research-ofi\features\{ticker}_features.parquet"
    df = pd.read_parquet(path)
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index('ts').sort_index()

    # Add stationary price features
    if 'microprice' in df.columns:
        df['microprice_ret'] = np.log(df['microprice'] / df['microprice'].shift(1))
    if 'microprice' in df.columns and 'vwap' in df.columns and 'spread' in df.columns:
        denom = df['spread'].replace(0, np.nan)
        df['vwap_deviation'] = (df['microprice'] - df['vwap']) / denom

    # Load regime labels and merge
    regime_path = rf"E:\quant-research-ofi\features\{ticker}_features_with_regimes.parquet"
    regimes = pd.read_parquet(regime_path)[['regime']]
    regimes.index = pd.to_datetime(regimes.index).tz_localize(None)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.join(regimes, how='left')

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

def run_ridge(X_train, y_train, X_test, y_test):
    if len(X_train) < 50 or len(X_test) < 5:
        return np.nan
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_sc, y_train)
    y_pred = ridge.predict(X_test_sc)
    return compute_ic(y_test, y_pred)

def run_fold(train_df, test_df, target, skip, regime_col):
    feature_cols = [f for f in SELECTED_FEATURES if f in train_df.columns]
    train_df = train_df.copy()
    test_df  = test_df.copy()

    train_df[feature_cols] = train_df[feature_cols].shift(1)
    test_df[feature_cols]  = test_df[feature_cols].shift(1)

    train_clean = train_df[feature_cols + [target, regime_col]].dropna()
    test_clean  = test_df[feature_cols + [target, regime_col]].dropna()
    test_nonoverlap = test_clean.iloc[::skip]

    if len(train_clean) < 100 or len(test_nonoverlap) < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Global — train all, test all
    ic_global = run_ridge(
        train_clean[feature_cols].values,
        train_clean[target].values,
        test_nonoverlap[feature_cols].values,
        test_nonoverlap[target].values
    )

    # Calm — train calm, test all
    train_calm = train_clean[train_clean[regime_col] == 0]
    ic_calm_all = run_ridge(
        train_calm[feature_cols].values,
        train_calm[target].values,
        test_nonoverlap[feature_cols].values,
        test_nonoverlap[target].values
    )

    # Calm matched — train calm, test calm only
    test_calm = test_nonoverlap[test_nonoverlap[regime_col] == 0]
    ic_calm_matched = run_ridge(
        train_calm[feature_cols].values,
        train_calm[target].values,
        test_calm[feature_cols].values,
        test_calm[target].values
    )

    # Stressed — train stressed, test all
    train_stressed = train_clean[train_clean[regime_col] == 2]
    ic_stressed_all = run_ridge(
        train_stressed[feature_cols].values,
        train_stressed[target].values,
        test_nonoverlap[feature_cols].values,
        test_nonoverlap[target].values
    )

    # Stressed matched — train stressed, test stressed only
    test_stressed = test_nonoverlap[test_nonoverlap[regime_col] == 2]
    ic_stressed_matched = run_ridge(
        train_stressed[feature_cols].values,
        train_stressed[target].values,
        test_stressed[feature_cols].values,
        test_stressed[target].values
    )

    return ic_global, ic_calm_all, ic_calm_matched, ic_stressed_all, ic_stressed_matched

def run_per_regime(ticker='SPY'):
    print(f"\n{'='*65}")
    print(f"PER-REGIME RIDGE MODELS — {ticker}")
    print(f"{'='*65}")

    df = load_data(ticker)

    regime_col = [c for c in df.columns if 'regime' in c.lower() or 'state' in c.lower()][0]
    print(f"Using regime column: {regime_col}")
    print(f"\nRegime distribution:")
    print(df[regime_col].value_counts())

    folds = get_wf_folds(df)
    print(f"\nWalk-forward folds: {len(folds)}")

    print(f"\n{'─'*65}")
    print(f"{'Horizon':<15} {'Model':<20} {'Mean IC':>8} {'NW T-stat':>10} {'Sig':>5}")
    print(f"{'─'*65}")

    all_results = []

    for target, skip in HORIZONS.items():
        if target not in df.columns:
            continue

        ics_global           = []
        ics_calm_all         = []
        ics_calm_matched     = []
        ics_stressed_all     = []
        ics_stressed_matched = []

        for train_df, test_df in folds:
            ic_g, ic_c_all, ic_c_matched, ic_s_all, ic_s_matched = run_fold(
                train_df, test_df, target, skip, regime_col)
            ics_global.append(ic_g)
            ics_calm_all.append(ic_c_all)
            ics_calm_matched.append(ic_c_matched)
            ics_stressed_all.append(ic_s_all)
            ics_stressed_matched.append(ic_s_matched)

        for label, ics in [
            ('Global',           ics_global),
            ('Calm-All',         ics_calm_all),
            ('Calm-Matched',     ics_calm_matched),
            ('Stressed-All',     ics_stressed_all),
            ('Stressed-Matched', ics_stressed_matched),
        ]:
            s = pd.Series(ics).dropna()
            if len(s) == 0:
                continue
            mean_ic  = s.mean()
            std_ic   = s.std()
            tstat_nw = newey_west_tstat(s)
            sig      = 'YES' if not np.isnan(tstat_nw) and abs(tstat_nw) > 2.0 else 'NO'

            print(f"{target:<15} {label:<20} {mean_ic:>8.4f} {tstat_nw:>10.3f} {sig:>5}")

            all_results.append({
                'horizon':     target,
                'model':       label,
                'mean_ic':     round(mean_ic, 4),
                'std_ic':      round(std_ic, 4) if not np.isnan(std_ic) else np.nan,
                'tstat_nw':    round(tstat_nw, 3) if not np.isnan(tstat_nw) else np.nan,
                'significant': sig
            })

        print(f"{'─'*65}")

    out = pd.DataFrame(all_results)
    out_path = rf"E:\quant-research-ofi\reports\{ticker}_per_regime_ic.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return out

if __name__ == "__main__":
    run_per_regime(ticker='SPY')
