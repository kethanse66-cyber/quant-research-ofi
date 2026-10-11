import pandas as pd
import numpy as np

# ── SYNTHETIC DATA ───────────────────────────────────────────
# SYNTHETIC ONLY — remove when using real Polygon data
np.random.seed(42)
n = 500
timestamps = pd.date_range(start='2024-01-02 09:30:00', periods=n, freq='10s')
prices = 100 + np.cumsum(np.random.randn(n) * 0.05)
df = pd.DataFrame({'timestamp': timestamps, 'price': prices})
df = df.set_index('timestamp')

# ── VERIFY INDEX FREQUENCY ───────────────────────────────────
print("=== INDEX FREQUENCY CHECK ===")
print(df.index.to_series().diff().value_counts().head())
print()

# ── TARGET VARIABLE ──────────────────────────────────────────
# Formula: log(price_t+n / price_t)
# Source: standard log return — Cont et al. (2014) and all microstructure research
# NOTE: replace 'price' with mid = (best_bid + best_ask) / 2 when using real Polygon data

def compute_log_returns(df, price_col='price'):
    df = df.copy()
    
    # Improvement 2: parameterized horizons — easy to add new ones later
    horizons = {'10s': 1, '1m': 6, '5m': 30}
    
    for name, steps in horizons.items():
        df[f'target_{name}'] = np.log(df[price_col].shift(-steps) / df[price_col])
    
    return df

df = compute_log_returns(df)

# ── TEST ─────────────────────────────────────────────────────
print("=== TARGET VARIABLE TEST ===")
print(f"Rows total: {len(df)}")
print(f"NaN in target_10s: {df['target_10s'].isna().sum()}")
print(f"NaN in target_1m:  {df['target_1m'].isna().sum()}")
print(f"NaN in target_5m:  {df['target_5m'].isna().sum()}")
print()
print("First 3 rows:")
print(df[['price', 'target_10s', 'target_1m', 'target_5m']].head(3).round(6))
print()
print("Last 3 rows (should be NaN — no future data):")
print(df[['price', 'target_10s', 'target_1m', 'target_5m']].tail(3).round(6))
