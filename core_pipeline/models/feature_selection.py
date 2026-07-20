# feature_selection.py
# Phase 5 ML Models — Feature Selection
# Remove correlated features and high-VIF features
# Reference: Lopez de Prado (2018) Ch.8

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# ── FEATURES ──────────────────────────────────────────────────────────────────

FEATURES = [
    'ofi', 'ofi_10s', 'ofi_30s', 'ofi_1m', 'ofi_5m', 'ofi_10m',
    'queue_imbalance', 'trade_imbalance',
    'spread', 'spread_change',
    'kyle_lambda', 'amihud',
    'realized_vol', 'ofi_norm'
]

# microprice_ret and vwap_deviation added as stationary versions
EXTRA_FEATURES = ['microprice_ret', 'vwap_deviation']

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_data(ticker='SPY'):
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

    return df

# ── STEP 1: CORRELATION FILTER ────────────────────────────────────────────────

def correlation_filter(df, features, threshold=0.8):
    """
    Remove features that are too correlated with each other.
    
    Why: If two features have correlation above 0.8 they are saying
    almost the same thing. Keeping both confuses the model and causes
    the condition number explosion we saw yesterday (10^13).
    
    Method: Build correlation matrix. For each pair above threshold,
    remove the one with higher average correlation to all others.
    Keep the one that is most 'unique'.
    
    Manual example:
    ofi and ofi_10s correlation = 0.97 → too high → remove one.
    ofi and spread correlation = 0.12 → fine → keep both.
    """
    available = [f for f in features if f in df.columns]
    df_clean = df[available].dropna()

    corr_matrix = df_clean.corr(method='spearman').abs()

    print(f"\nCorrelation matrix ({len(available)} features):")
    print(corr_matrix.round(2).to_string())

    # Find pairs above threshold
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = set()
    for col in upper.columns:
        high_corr = upper[col][upper[col] > threshold].index.tolist()
        for corr_col in high_corr:
            # Drop the one with higher average correlation
            avg_corr_col = corr_matrix[col].mean()
            avg_corr_corr_col = corr_matrix[corr_col].mean()
            if avg_corr_col > avg_corr_corr_col:
                to_drop.add(col)
            else:
                to_drop.add(corr_col)

    survived = [f for f in available if f not in to_drop]

    print(f"\nCorrelation filter (threshold={threshold}):")
    print(f"  Dropped  : {sorted(to_drop)}")
    print(f"  Survived : {survived}")

    return survived, corr_matrix

# ── STEP 2: VIF FILTER ────────────────────────────────────────────────────────

def vif_filter(df, features, threshold=10.0):
    """
    Remove features with high Variance Inflation Factor.
    
    Why: VIF measures how much one feature can be explained by all others.
    VIF = 1/(1 - R²) where R² is from regressing that feature on all others.
    
    VIF < 5: fine
    VIF 5-10: borderline
    VIF > 10: severe multicollinearity — remove
    
    Manual example:
    ofi_1m regressed on all other features gives R²=0.99
    VIF = 1/(1-0.99) = 100 → severe → remove ofi_1m
    
    Reference: Lopez de Prado (2018) Ch.8
    """
    available = [f for f in features if f in df.columns]
    df_clean = df[available].dropna()
    X = df_clean.values

    print(f"\nVIF scores (threshold={threshold}):")

    survived = list(available)

    # Iteratively remove highest VIF until all below threshold
    while True:
        X_current = df_clean[survived].dropna().values
        vif_scores = []
        for i in range(X_current.shape[1]):
            try:
                vif = variance_inflation_factor(X_current, i)
            except:
                vif = np.nan
            vif_scores.append(vif)

        vif_df = pd.DataFrame({
            'feature': survived,
            'vif': vif_scores
        }).sort_values('vif', ascending=False)

        print(vif_df.to_string(index=False))

        max_vif = vif_df['vif'].max()
        if max_vif <= threshold or len(survived) <= 3:
            break

        # Remove feature with highest VIF
        worst = vif_df.iloc[0]['feature']
        survived.remove(worst)
        print(f"\n  Removing: {worst} (VIF={max_vif:.1f})")
        print(f"  Remaining: {survived}\n")

    print(f"\nVIF filter complete. Final features: {survived}")
    return survived, vif_df

# ── STEP 3: RERUN RIDGE WITH SELECTED FEATURES ────────────────────────────────

def quick_ic_check(df, features, target='fwd_ret_5m'):
    """
    Quick IC check on selected features.
    Uses simple train/test split — not full walk-forward.
    Purpose: verify condition number dropped after feature selection.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    available = [f for f in features if f in df.columns]
    df_model = df[available + [target]].dropna().copy()

    # Shift features by 1
    df_model[available] = df_model[available].shift(1)
    df_model = df_model.dropna()

    # Simple 80/20 split
    split = int(len(df_model) * 0.8)
    train = df_model.iloc[:split]
    test  = df_model.iloc[split::30]  # non-overlapping

    X_train = train[available].values
    y_train = train[target].values
    X_test  = test[available].values
    y_test  = test[target].values

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    cond = np.linalg.cond(X_train_sc)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_sc, y_train)
    y_pred = ridge.predict(X_test_sc)

    ic, _ = stats.spearmanr(y_test, y_pred)

    print(f"\nQuick IC check with selected features:")
    print(f"  Features used   : {len(available)}")
    print(f"  Condition number: {cond:.2f}  (was 10^13 before)")
    print(f"  IC              : {ic:.4f}")
    print(f"  Condition number {'FIXED' if cond < 1000 else 'STILL HIGH'}")

    return ic, cond

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_feature_selection(ticker='SPY'):
    print(f"\n{'='*60}")
    print(f"FEATURE SELECTION — {ticker}")
    print(f"{'='*60}")

    df = load_data(ticker)

    all_features = FEATURES + EXTRA_FEATURES

    # Step 1: Correlation filter
    corr_survived, corr_matrix = correlation_filter(df, all_features, threshold=0.8)

    # Step 2: VIF filter
    final_features, vif_df = vif_filter(df, corr_survived, threshold=10.0)

    # Step 3: Quick IC check
    ic, cond = quick_ic_check(df, final_features)

    # Save results
    results = pd.DataFrame({
        'feature': final_features,
        'selected': True
    })
    out_path = rf"E:\quant-research-ofi\reports\{ticker}_selected_features.csv"
    results.to_csv(out_path, index=False)

    print(f"\n{'─'*60}")
    print(f"FEATURE SELECTION COMPLETE — {ticker}")
    print(f"  Started with  : {len(all_features)} features")
    print(f"  After corr    : {len(corr_survived)} features")
    print(f"  After VIF     : {len(final_features)} features")
    print(f"  Final features: {final_features}")
    print(f"  Condition #   : {cond:.2f}")
    print(f"  Quick IC      : {ic:.4f}")
    print(f"  Saved to      : {out_path}")
    print(f"{'─'*60}")

    return final_features


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    selected = run_feature_selection(ticker='SPY')
    print(f"\nFinal selected features for all downstream models:")
    for f in selected:
        print(f"  {f}")
