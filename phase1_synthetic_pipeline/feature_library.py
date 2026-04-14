# feature_library.py
# Builds microprice, spread_change, and ofi_norm features.
# Reference: Cont, Kukanov & Stoikov (2014)

import numpy as np
import pandas as pd

np.random.seed(42)
N = 500

# ── SYNTHETIC ONLY — replace with real Polygon data in Phase 2 ──────────────
def generate_synthetic_book(n, signal_strength=0.3):
    bid_size = np.random.randint(100, 1000, size=n)
    ask_size = np.random.randint(100, 1000, size=n)
    ofi_raw = np.diff(bid_size) - np.diff(ask_size)
    ofi_signal = np.append(0, ofi_raw)
    ofi_norm = ofi_signal / (np.abs(ofi_signal).max() + 1e-9)
    ofi_lagged = np.roll(ofi_norm, 1)
    ofi_lagged[0] = 0
    noise = np.random.randn(n) * 0.05
    price_innovations = signal_strength * ofi_lagged + (1 - signal_strength) * noise
    bid_prices = np.round(100 + np.cumsum(price_innovations) * 0.1, 4)
    ask_prices = np.round(bid_prices + np.random.uniform(0.01, 0.05, size=n), 4)
    weighted_mid = (bid_size * ask_prices + ask_size * bid_prices) / (bid_size + ask_size)
    return pd.DataFrame({
        "bid_price":    bid_prices,
        "ask_price":    ask_prices,
        "bid_size":     bid_size,
        "ask_size":     ask_size,
        "spread":       np.round(ask_prices - bid_prices, 4),
        "mid_price":    np.round((bid_prices + ask_prices) / 2, 4),
        "weighted_mid": np.round(weighted_mid, 4),
    })

def compute_ofi(df):
    bid_change = df["bid_size"].diff()
    ask_change = df["ask_size"].diff()
    bid_price_change = df["bid_price"] != df["bid_price"].shift(1)
    ask_price_change = df["ask_price"] != df["ask_price"].shift(1)
    bid_change[bid_price_change] = df["bid_size"][bid_price_change]
    ask_change[ask_price_change] = df["ask_size"][ask_price_change]
    return bid_change - ask_change
# ── END SYNTHETIC ONLY ───────────────────────────────────────────────────────

df = generate_synthetic_book(N, signal_strength=0.3)
df["ofi"] = compute_ofi(df)

# ── FEATURE 1: MICROPRICE ────────────────────────────────────────────────────
df["microprice"] = (
    (df["bid_size"] * df["ask_price"]) + (df["ask_size"] * df["bid_price"])
) / (df["bid_size"] + df["ask_size"])
df["microprice"] = df["microprice"].round(4)

# ── FEATURE 2: SPREAD CHANGE ─────────────────────────────────────────────────
df["spread_change"] = df["spread"].diff()
df["spread_change"] = df["spread_change"].round(4)

# ── FEATURE 3: OFI NORM ──────────────────────────────────────────────────────
df["ofi_norm"] = df["ofi"] / df["ofi"].rolling(20).std()

# ── TEST ─────────────────────────────────────────────────────────────────────
print("=== Feature Library — Validation ===")
print(f"microprice row 1  : Expected close to mid_price | Got {df['microprice'].iloc[1]}")
print(f"mid_price row 1   : {df['mid_price'].iloc[1]}")
print(f"spread_change row 1: {df['spread_change'].iloc[1]}")
print(f"ofi_norm row 20   : {df['ofi_norm'].iloc[20]:.4f}")
print(f"ofi_norm NaNs     : {df['ofi_norm'].isna().sum()} (expect 19)")
print("\n=== Columns in df ===")
print(df.columns.tolist())
