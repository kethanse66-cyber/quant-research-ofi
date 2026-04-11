# tick_cleaner.py
# Cleans raw tick data — removes bad prices, fills gaps, saves to Parquet.
 
import numpy as np
import pandas as pd
 
np.random.seed(42)
 
 
def generate_raw_ticks(n=100):
    """Fake tick data with typical real-world data quality problems.
 
    Injects: duplicate timestamps, missing prices, zero price,
    negative price, and one extreme volume (fat-finger).
    """
    dates = pd.date_range(start="2024-01-01 09:30:00", periods=n, freq="1s")
    df = pd.DataFrame({
        "timestamp": dates,
        "price":     100 + np.cumsum(np.random.randn(n) * 0.01),
        "volume":    np.random.randint(100, 1000, n),
    })
    duplicates = df.sample(5, random_state=42)
    df = pd.concat([df, duplicates]).reset_index(drop=True)
    missing_idx = np.random.choice(df.index, 5, replace=False)
    df.loc[missing_idx, "price"] = np.nan
    df.loc[2, "price"]  = 0
    df.loc[5, "price"]  = -10
    df.loc[7, "volume"] = 999999
    return df
 
 
def clean_ticks(df, output_path="clean_ticks.parquet"):
    """Full cleaning pipeline — runs steps in this exact order:
 
    1. Drop duplicate timestamps (keep first)
    2. Remove zero and negative prices — bad data, not gaps
    3. Forward-fill remaining missing prices — genuine gaps
    4. Remove zero volume and extreme volume (fat-finger guard)
    5. Resample to regular 1-second grid
    6. Convert timestamps to UTC
 
    Order matters: bad prices must be removed before ffill.
    If ffill runs first, bad values spread forward and survive.
    """
    audit = {"rows_in": len(df)}
 
    # step 1 — duplicates
    before = len(df)
    df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)
    audit["duplicates_removed"] = before - len(df)
 
    # step 2 — bad prices before ffill
    before = len(df)
    df = df[df["price"] > 0].reset_index(drop=True)
    audit["bad_prices_removed"] = before - len(df)
 
    # step 3 — forward fill genuine gaps
    missing_before = df["price"].isna().sum()
    df["price"] = df["price"].ffill()
    audit["prices_ffilled"] = int(missing_before)
 
    # step 4 — bad volume
    before = len(df)
    vol_mean = df["volume"].mean()
    vol_std  = df["volume"].std()
    vol_cap  = vol_mean + 5 * vol_std
    df = df[(df["volume"] > 0) & (df["volume"] < vol_cap)].reset_index(drop=True)
    audit["bad_volume_removed"] = before - len(df)
 
    # step 5 — regular 1s grid
    df = df.set_index("timestamp")
    df = df.resample("1s").ffill()
    df = df.reset_index()
 
    # step 6 — UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
 
    audit["rows_out"] = len(df)
    df.to_parquet(output_path, index=False)
    return df, audit
 
 
if __name__ == "__main__":
    raw = generate_raw_ticks(n=100)
    print(f"Raw rows  : {len(raw)}")
    print(f"Sample bad prices:\n{raw.loc[[2,5], ['timestamp','price','volume']].to_string()}")
 
    clean, audit = clean_ticks(raw, output_path="clean_ticks.parquet")
 
    print("\n=== Audit Log ===")
    for k, v in audit.items():
        print(f"  {k:<25}: {v}")
 
    print("\n=== Clean Data Preview ===")
    print(clean.head(10))
    print("\nclean_ticks.parquet saved")
 
