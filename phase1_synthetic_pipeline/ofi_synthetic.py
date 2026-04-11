# ofi_synthetic.py
# Generates synthetic order book data and validates the OFI pipeline.
# Reference: Cont, Kukanov & Stoikov (2014)

import numpy as np
import pandas as pd

np.random.seed(42)
N = 500


def generate_synthetic_book(n, signal_strength=0.3):
    """Fake order book data with a small OFI signal baked in."""

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
    """Best-touch OFI with price-change reset (Cont et al. 2014)."""

    bid_change = df["bid_size"].diff()
    ask_change = df["ask_size"].diff()

    # if best price changes → previous queue is gone
    bid_price_change = df["bid_price"] != df["bid_price"].shift(1)
    ask_price_change = df["ask_price"] != df["ask_price"].shift(1)

    bid_change[bid_price_change] = df["bid_size"][bid_price_change]
    ask_change[ask_price_change] = df["ask_size"][ask_price_change]

    return bid_change - ask_change


def add_rolling_features(df, windows=(5, 10)):
    """Add rolling OFI features."""

    for w in windows:
        df[f"ofi_mean_{w}"] = df["ofi"].rolling(w).mean()
        df[f"ofi_std_{w}"]  = df["ofi"].rolling(w).std()
        df[f"ofi_sum_{w}"]  = df["ofi"].rolling(w).sum()

    return df


def compute_forward_return(df, horizon=1):
    """Forward weighted mid return."""
    return df["weighted_mid"].pct_change().shift(-horizon)


def compute_ic(signal, forward_return):
    """Spearman IC."""
    combined = pd.concat([signal, forward_return], axis=1).dropna()
    return combined.iloc[:, 0].corr(combined.iloc[:, 1], method="spearman")


if __name__ == "__main__":
    df = generate_synthetic_book(N, signal_strength=0.3)

    df["ofi"] = compute_ofi(df)

    df = add_rolling_features(df, windows=(5, 10))

    df["future_return"] = compute_forward_return(df, horizon=1)

    print("=== Synthetic Book — First 5 Rows ===")
    print(df[["bid_price", "ask_price", "bid_size", "ask_size",
              "spread", "mid_price", "weighted_mid"]].head())

    print("\n=== OFI Summary ===")
    ofi_clean = df["ofi"].dropna()
    print(f"Mean OFI        : {ofi_clean.mean():.2f}")
    print(f"Std OFI         : {ofi_clean.std():.2f}")
    print(f"% buy pressure  : {(ofi_clean > 0).mean() * 100:.1f}%")

    print("\n=== IC Validation (Spearman) ===")
    ic_raw = compute_ic(df["ofi"], df["future_return"])
    ic_5   = compute_ic(df["ofi_mean_5"], df["future_return"])
    ic_10  = compute_ic(df["ofi_mean_10"], df["future_return"])

    print(f"IC raw OFI      : {ic_raw:.4f}")
    print(f"IC 5-bar mean   : {ic_5:.4f}")
    print(f"IC 10-bar mean  : {ic_10:.4f}")

    if ic_raw > 0.05:
        print("\nPASS — injected signal detectable. Pipeline correctly aligned.")
    else:
        print("\nWARN — IC near zero. Check lag alignment or look-ahead bias.")
