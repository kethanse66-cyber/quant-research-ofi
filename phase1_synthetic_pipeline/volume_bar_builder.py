import pandas as pd
import numpy as np

# Load clean tick data from Day 4
df = pd.read_parquet("clean_ticks.parquet")

# VOLUME BARS
volume_threshold = 500
df["cumulative_volume"] = df["volume"].cumsum()
df["bar_id"] = (df["cumulative_volume"] / volume_threshold).astype(int)

volume_bars = df.groupby("bar_id").apply(
    lambda x: pd.Series({
        "timestamp": x["timestamp"].iloc[-1],
        "vwap": (x["price"] * x["volume"]).sum() / x["volume"].sum(),
        "total_volume": x["volume"].sum()
    })
).reset_index(drop=True)

# TIME BARS — for comparison only, not used in pipeline
df = df.set_index("timestamp")
time_bars = df["price"].resample("1min").ohlc()
time_bars["volume"] = df["volume"].resample("1min").sum()

# Compare
print("Volume bar stats:")
print(volume_bars["total_volume"].describe())
print("\nTime bar stats:")
print(time_bars["volume"].describe())

# Save
volume_bars.to_parquet("volume_bars.parquet", index=False)
print("\nvolume_bars.parquet saved")
