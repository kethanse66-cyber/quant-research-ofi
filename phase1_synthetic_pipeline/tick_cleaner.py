import pandas as pd
import numpy as np

# -------------------------
# create synthetic data
# -------------------------
np.random.seed(42)

dates = pd.date_range(
    start="2024-01-01 09:30:00",
    periods=100,
    freq="1s"
)

df = pd.DataFrame({
    "timestamp": dates,
    "price": 100 + np.cumsum(np.random.randn(100) * 0.01),
    "volume": np.random.randint(100, 1000, 100)
})

# -------------------------
# inject duplicates
# -------------------------
duplicates = df.sample(5, random_state=42)
df = pd.concat([df, duplicates]).reset_index(drop=True)

# -------------------------
# inject missing values
# -------------------------
missing_idx = np.random.choice(df.index, 5, replace=False)
df.loc[missing_idx, "price"] = np.nan

# -------------------------
# inject bad prices
# -------------------------
df.loc[2, "price"] = 0
df.loc[5, "price"] = -10

# -------------------------
# drop duplicates
# -------------------------
df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)

# -------------------------
# drop zero volume
# -------------------------
df = df[df["volume"] > 0].reset_index(drop=True)

# -------------------------
# fill missing price
# -------------------------
df["price"] = df["price"].ffill()

# -------------------------
# drop bad price
# -------------------------
df = df[df["price"] > 0].reset_index(drop=True)

# -------------------------
# convert timestamp
# -------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"])

# -------------------------
# fill missing timestamps
# -------------------------
df = df.set_index("timestamp")

df = df.resample("1S").ffill()

df = df.reset_index()

# -------------------------
# convert to utc
# -------------------------
df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

# -------------------------
# save parquet
# -------------------------
df.to_parquet("clean_ticks.parquet", index=False)

print(df.head())
print("tick cleaning complete")
