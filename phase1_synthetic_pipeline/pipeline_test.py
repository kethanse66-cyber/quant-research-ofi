import pandas as pd
import numpy as np
import time
import os

# ============================================================
# PIPELINE TEST — Phase 1 End-to-End (Full Version)
# ============================================================

def generate_synthetic_data(n=1000):
    # SYNTHETIC ONLY — remove when using real Polygon data
    np.random.seed(42)
    timestamps = pd.date_range(start='2024-01-02 09:30:00', periods=n, freq='10s')

    # Clustered volatility — GARCH-like vol switching
    vol = np.ones(n) * 0.01
    for i in range(1, n):
        shock = np.random.randn()
        vol[i] = np.sqrt(0.0001 + 0.1 * (shock * vol[i-1])**2 + 0.85 * vol[i-1]**2)

    # Price with jumps
    returns = np.random.randn(n) * vol
    jump_mask = np.random.rand(n) < 0.01  # 1% chance of jump each row
    returns[jump_mask] += np.random.choice([-0.05, 0.05], jump_mask.sum())
    best_bid = 100 + np.cumsum(returns)

    # Spread widens in high vol regimes
    base_spread = np.random.uniform(0.01, 0.03, n)
    spread_multiplier = 1 + 5 * vol / vol.max()
    best_ask = best_bid + base_spread * spread_multiplier

    # Volume bursts correlated with vol
    base_volume = np.random.randint(100, 1000, n).astype(float)
    volume = base_volume * (1 + 10 * vol / vol.max())

    bid_size = np.random.randint(100, 1000, n).astype(float)
    ask_size = np.random.randint(100, 1000, n).astype(float)
    price    = (best_bid + best_ask) / 2 + np.random.randn(n) * 0.001

    df = pd.DataFrame({
        'timestamp': timestamps,
        'best_bid':  best_bid,
        'best_ask':  best_ask,
        'bid_size':  bid_size,
        'ask_size':  ask_size,
        'price':     price,
        'volume':    volume
    })
    df.set_index('timestamp', inplace=True)
    return df

def compute_ofi(df):
    """
    Full OFI formula — Cont, Kukanov & Stoikov (2014)
    Handles: size changes + price changes + queue depletion
    
    OFI_t = delta_bid_contribution - delta_ask_contribution
    
    Bid contribution:
      +bid_size if bid price rose (new queue added)
      -bid_size if bid price fell (queue depleted)
      +delta_bid_size if bid price unchanged
    
    Ask contribution:
      +ask_size if ask price fell (new queue added)
      -ask_size if ask price rose (queue depleted)
      +delta_ask_size if ask price unchanged
    """
    bid_price_up   = df['best_bid'] > df['best_bid'].shift(1)
    bid_price_down = df['best_bid'] < df['best_bid'].shift(1)
    bid_unchanged  = df['best_bid'] == df['best_bid'].shift(1)

    ask_price_down = df['best_ask'] < df['best_ask'].shift(1)
    ask_price_up   = df['best_ask'] > df['best_ask'].shift(1)
    ask_unchanged  = df['best_ask'] == df['best_ask'].shift(1)

    delta_bid_size = df['bid_size'].diff()
    delta_ask_size = df['ask_size'].diff()

    bid_contrib = np.where(bid_price_up,    df['bid_size'],
                  np.where(bid_price_down, -df['bid_size'],
                  np.where(bid_unchanged,   delta_bid_size, 0)))

    ask_contrib = np.where(ask_price_down,  df['ask_size'],
                  np.where(ask_price_up,   -df['ask_size'],
                  np.where(ask_unchanged,   delta_ask_size, 0)))

    return bid_contrib - ask_contrib

def compute_features(df):
    out = df.copy()

    # --- Full OFI ---
    out['ofi'] = compute_ofi(df)

    # --- OFI horizons ---
    for label, window in [('ofi_10s',1),('ofi_30s',3),('ofi_1m',6),
                           ('ofi_5m',30),('ofi_10m',60)]:
        out[label] = out['ofi'].rolling(window).sum()

    # --- Spread and spread_change ---
    out['spread']        = out['best_ask'] - out['best_bid']
    out['spread_change'] = out['spread'].diff()

    # --- Microprice ---
    out['microprice'] = (out['bid_size'] * out['best_ask'] +
                         out['ask_size'] * out['best_bid']) / \
                        (out['bid_size'] + out['ask_size'])

    # --- VWAP ---
    out['vwap'] = (out['price'] * out['volume']).cumsum() / out['volume'].cumsum()

    # --- Queue imbalance ---
    total = out['bid_size'] + out['ask_size']
    out['queue_imbalance'] = np.where(total > 0,
                                      (out['bid_size'] - out['ask_size']) / total,
                                      0.0)

    # --- Trade imbalance ---
    mid  = (out['best_bid'] + out['best_ask']) / 2
    sign = np.sign(out['price'] - mid)
    out['trade_imbalance'] = sign.rolling(6).mean()

    # --- Kyle's Lambda ---
    ret        = out['price'].pct_change()
    signed_vol = sign * out['volume']
    out['kyle_lambda'] = (ret.rolling(20).cov(signed_vol) /
                          signed_vol.rolling(20).var().replace(0, np.nan))

    # --- Amihud ---
    out['amihud'] = ret.abs() / out['volume'].replace(0, np.nan)

    # --- Realized vol ---
    out['realized_vol'] = ret.rolling(20).std()

    # --- OFI norm ---
    ofi_std = out['ofi'].rolling(20).std().replace(0, np.nan)
    out['ofi_norm'] = out['ofi'] / ofi_std

    return out

def add_targets_and_lags(df):
    out = df.copy()

    # Target variables — log forward returns
    for label, shift in [('ret_10s',1),('ret_1m',6),('ret_5m',30)]:
        out[label] = np.log(out['price'].shift(-shift) / out['price'])

    # Lag all features by 1 — prevent look-ahead bias
    feature_cols = [
        'ofi','ofi_10s','ofi_30s','ofi_1m','ofi_5m','ofi_10m',
        'queue_imbalance','trade_imbalance',
        'spread','spread_change','microprice','vwap',
        'kyle_lambda','amihud','realized_vol','ofi_norm'
    ]
    for col in feature_cols:
        out[col] = out[col].shift(1)

    return out, feature_cols

def run_unit_tests(df, feature_cols):
    print("\n[UNIT TESTS]")

    # Spread always positive
    assert df['spread'].min() > 0, "FAIL — negative spread found"
    print("  PASS — spread > 0 always")

    # Queue imbalance in [-1, +1]
    assert df['queue_imbalance'].min() >= -1.0, "FAIL — queue_imbalance below -1"
    assert df['queue_imbalance'].max() <= 1.0,  "FAIL — queue_imbalance above +1"
    print("  PASS — queue_imbalance in [-1, +1]")

    # Trade imbalance in [-1, +1]
    ti = df['trade_imbalance'].dropna()
    assert ti.min() >= -1.0, "FAIL — trade_imbalance below -1"
    assert ti.max() <= 1.0,  "FAIL — trade_imbalance above +1"
    print("  PASS — trade_imbalance in [-1, +1]")

    # No future leakage — target must not appear in features
    assert 'ret_1m' not in feature_cols, "FAIL — target variable in feature list"
    print("  PASS — target not in feature list")

    # Features are lagged — first row of features must be NaN
    assert pd.isna(df[feature_cols].iloc[0]['ofi']), "FAIL — first row of ofi not NaN after shift"
    print("  PASS — features lagged, first row is NaN")

    # Realized vol always non-negative
    rv = df['realized_vol'].dropna()
    assert rv.min() >= 0, "FAIL — negative realized vol"
    print("  PASS — realized_vol >= 0 always")

def run_ic_analysis(df, feature_cols):
    """Quick IC check — rank correlation between each feature and ret_1m"""
    print("\n[IC ANALYSIS — rank correlation vs ret_1m]")
    df_clean = df[feature_cols + ['ret_1m']].dropna()
    ics = {}
    for col in feature_cols:
        ic = df_clean[col].corr(df_clean['ret_1m'], method='spearman')
        ics[col] = round(ic, 4)
    ic_series = pd.Series(ics).sort_values(ascending=False)
    print(ic_series.to_string())
    return ic_series

def run_ridge_baseline(df, feature_cols):
    """Ridge regression baseline — OFI features vs ret_1m"""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    print("\n[RIDGE BASELINE]")
    df_clean = df[feature_cols + ['ret_1m']].dropna()
    X = df_clean[feature_cols].values
    y = df_clean['ret_1m'].values

    # Walk-forward: train on first 70%, test on last 30%
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    ic_oos = pd.Series(preds).corr(pd.Series(y_test), method='spearman')
    r2     = r2_score(y_test, preds)
    print(f"  OOS IC (Spearman) : {ic_oos:.4f}")
    print(f"  OOS R²            : {r2:.4f}")
    print(f"  Train rows        : {len(X_train)}")
    print(f"  Test rows         : {len(X_test)}")

def save_parquet(df, path='SPY_features.parquet'):
    df.to_parquet(path, engine='pyarrow')
    return path

def audit_lookahead(df, feature_cols):
    first_valid = df[feature_cols].dropna(how='all').index[0]
    print(f"\n[AUDIT]")
    print(f"  First valid prediction row : {first_valid}")
    print(f"  Features used              : {len(feature_cols)}")
    print(f"  All features lagged by shift(1) — no look-ahead bias")

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    start = time.time()
    print("="*55)
    print("PIPELINE TEST — Phase 1 Synthetic (Full Version)")
    print("="*55)

    print("\n[1] Generating synthetic data...")
    df_raw = generate_synthetic_data(n=1000)
    print(f"    Input rows : {len(df_raw)}")

    print("\n[2] Computing features...")
    df_feat = compute_features(df_raw)

    print("\n[3] Adding targets and lagging features...")
    df_final, feature_cols = add_targets_and_lags(df_feat)
    print(f"    Output rows   : {len(df_final)}")
    print(f"    Feature count : {len(feature_cols)}")

    audit_lookahead(df_final, feature_cols)

    run_unit_tests(df_final, feature_cols)

    run_ic_analysis(df_final, feature_cols)

    run_ridge_baseline(df_final, feature_cols)

    print("\n[4] Saving to Parquet...")
    path = save_parquet(df_final)
    size_mb = os.path.getsize(path) / (1024*1024)
    print(f"    File : {path}")
    print(f"    Size : {size_mb:.3f} MB")

    print("\n[5] Reloading Parquet to verify...")
    df_check = pd.read_parquet(path)
    print(f"    Reloaded rows    : {len(df_check)}")
    print(f"    Reloaded columns : {len(df_check.columns)}")

    elapsed = time.time() - start
    print(f"\n{'='*55}")
    print(f"PIPELINE COMPLETE — {elapsed:.2f} seconds")
    print(f"{'='*55}")
