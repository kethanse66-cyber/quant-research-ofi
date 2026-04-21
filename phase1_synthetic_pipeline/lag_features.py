import pandas as pd
import numpy as np

# SYNTHETIC ONLY — remove when using real Polygon data
np.random.seed(42)
n = 500
timestamps = pd.date_range(start='2024-01-02 09:30:00', periods=n, freq='10s')
prices     = 100 + np.cumsum(np.random.randn(n) * 0.05)
bid_sizes  = np.random.randint(100, 1000, n)
ask_sizes  = np.random.randint(100, 1000, n)
volumes    = np.random.randint(50, 500, n)

df = pd.DataFrame({
    'price'    : prices,
    'bid_size' : bid_sizes,
    'ask_size' : ask_sizes,
    'volume'   : volumes
}, index=timestamps)
df.index.name = 'timestamp'

# ── FEATURES ─────────────────────────────────────────────────
# SYNTHETIC ONLY — real OFI from ofi_full.py used in Phase 2
df['ofi']             = df['bid_size'].diff() - df['ask_size'].diff()
df['queue_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'])

# shift(1) on rolling std — avoids using current row inside window
df['ofi_norm']        = df['ofi'] / df['ofi'].rolling(20).std().shift(1)
df['volume_norm']     = df['volume'] / df['volume'].rolling(20).mean().shift(1)
# NOTE: real spread = ask_price - bid_price
# SYNTHETIC ONLY — no price levels in synthetic data

# ── TARGET VARIABLE ──────────────────────────────────────────
df['target_1m'] = np.log(df['price'].shift(-6) / df['price'])

# ── LAG FEATURES ─────────────────────────────────────────────
# Formula: df.shift(lag) — use previous row values as features
# Source: standard no-lookahead feature construction in time series ML
# Reason: model at time T must only see data from T-1 and before

feature_cols = ['ofi', 'queue_imbalance', 'ofi_norm', 'volume_norm']

def create_lag_features(df, feature_cols, lags=[1, 2, 3]):
    df = df.copy()
    for col in feature_cols:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

df = create_lag_features(df, feature_cols)

# ── CLEAN MODEL DATAFRAME ────────────────────────────────────
lag_cols = [c for c in df.columns if 'lag' in c]
model_df = df.dropna(subset=lag_cols + ['target_1m'])

first_valid = model_df.index[0]

# ── TEST ─────────────────────────────────────────────────────
print("=== LAG FEATURES TEST ===")
print(f"Total rows: {len(df)}")
print(f"Rows after dropna: {len(model_df)}")
print(f"First valid prediction row: {first_valid}")
print()
print("Lag columns created:")
print(lag_cols)
print()
print("First 4 rows — ofi and lags side by side:")
print(model_df[['ofi', 'ofi_lag1', 'ofi_lag2', 'ofi_lag3', 'target_1m']].head(4).round(4))
print()
print("NaN count per lag column (should be 0 after dropna):")
for col in lag_cols:
    print(f"  {col}: {model_df[col].isna().sum()} NaNs")
