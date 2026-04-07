import numpy as np
import pandas as pd

# reproducibility
np.random.seed(42)

# -----------------------------
# Synthetic order book sizes
# -----------------------------
n = 100

bid_size = np.random.randint(100, 1000, size=n)
ask_size = np.random.randint(100, 1000, size=n)

print("Bid size:", bid_size[:5])
print("Ask size:", ask_size[:5])

# -----------------------------
# OFI calculation
# -----------------------------
delta_bid = np.diff(bid_size)
delta_ask = np.diff(ask_size)

ofi = delta_bid - delta_ask

print("\nOFI first 10 values:", ofi[:10])
print("Mean OFI:", round(ofi.mean(), 2))
print("Std OFI:", round(ofi.std(), 2))
print("% Buy pressure:", round((ofi > 0).mean() * 100, 2), "%")

# -----------------------------
# Synthetic prices
# -----------------------------
bid_prices = np.round(100 + np.random.randn(n).cumsum() * 0.1, 2)
ask_prices = np.round(bid_prices + np.random.uniform(0.01, 0.05, size=n), 2)

print("\nBid price:", bid_prices[:5])
print("Ask price:", ask_prices[:5])

# -----------------------------
# Spread calculation
# -----------------------------
spread = np.round(ask_prices - bid_prices, 4)

print("\nSpread first 10:", spread[:10])
print("Mean spread:", round(spread.mean(), 4))
print("Std spread:", round(spread.std(), 4))

# -----------------------------
# Create DataFrame
# -----------------------------
df = pd.DataFrame({
    "bid_price": bid_prices,
    "ask_price": ask_prices,
    "bid_size": bid_size,
    "ask_size": ask_size,
    "spread": spread,
    "ofi": np.append(np.nan, ofi)
})

# -----------------------------
# Rolling features
# -----------------------------
df["ofi_roll_mean_10"] = df["ofi"].rolling(10).mean()
df["ofi_roll_sum_10"] = df["ofi"].rolling(10).sum()
df["ofi_roll_std_10"] = df["ofi"].rolling(10).std()

df["ofi_roll_mean_5"] = df["ofi"].rolling(5).mean()

# -----------------------------
# Output
# -----------------------------
print("\nData preview:")
print(df.head(15))
# -----------------------------
# Mid price + returns
# -----------------------------
df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
df["mid_return"] = df["mid_price"].pct_change()
df["future_mid_return"] = df["mid_return"].shift(-1)

# -----------------------------
# IC calculation
# -----------------------------
ic = df["ofi"].corr(df["future_mid_return"])
print("OFI IC:", ic)

print(df[["ofi","mid_price","mid_return","future_mid_return"]].head(10))
