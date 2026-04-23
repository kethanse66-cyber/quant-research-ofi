import pandas as pd
import numpy as np
import os
import time

# ── SYNTHETIC ONLY — remove when using real Polygon data ──────────────────────
def make_synthetic_features(n=100_000):
    np.random.seed(42)
    timestamps = pd.date_range(
        start="2024-01-02 09:30:00", periods=n, freq="10s", tz="UTC"
    )
    df = pd.DataFrame({
        "timestamp":       timestamps,
        "ofi":             np.random.randn(n),
        "ofi_10s":         np.random.randn(n),
        "ofi_30s":         np.random.randn(n),
        "ofi_1m":          np.random.randn(n),
        "ofi_5m":          np.random.randn(n),
        "ofi_10m":         np.random.randn(n),
        "queue_imbalance": np.random.uniform(-1, 1, n),
        "trade_imbalance": np.random.uniform(-1, 1, n),
        "spread":          np.random.uniform(0.01, 0.05, n),
        "spread_change":   np.random.randn(n) * 0.001,
        "microprice":      100 + np.random.randn(n) * 0.1,
        "vwap":            100 + np.random.randn(n) * 0.1,
        "kyle_lambda":     np.random.uniform(0.001, 0.01, n),
        "amihud":          np.random.uniform(0.0001, 0.001, n),
        "realized_vol":    np.random.uniform(0.01, 0.03, n),
        "ofi_norm":        np.random.randn(n),
    })
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    return df
# ── END SYNTHETIC ONLY ────────────────────────────────────────────────────────


def benchmark_csv_vs_parquet(df, ticker, output_dir="parquet_data"):
    os.makedirs(output_dir, exist_ok=True)

    csv_path     = os.path.join(output_dir, f"{ticker}_features.csv")
    parquet_path = os.path.join(output_dir, f"{ticker}_features.parquet")

    # ── CSV write ──
    try:
        t0 = time.perf_counter()
        df.to_csv(csv_path, index=False)
        csv_write_time = time.perf_counter() - t0
    except Exception as e:
        print(f"CSV write failed: {e}")
        return None

    # ── CSV read ──
    try:
        t0 = time.perf_counter()
        pd.read_csv(csv_path)
        csv_read_time = time.perf_counter() - t0
    except Exception as e:
        print(f"CSV read failed: {e}")
        return None

    csv_size_mb = os.path.getsize(csv_path) / (1024 * 1024)

    # ── Parquet write with snappy compression ──
    try:
        t0 = time.perf_counter()
        df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")
        parquet_write_time = time.perf_counter() - t0
    except Exception as e:
        print(f"Parquet write failed: {e}")
        return None

    # ── Parquet read ──
    try:
        t0 = time.perf_counter()
        pd.read_parquet(parquet_path, engine="pyarrow")
        parquet_read_time = time.perf_counter() - t0
    except Exception as e:
        print(f"Parquet read failed: {e}")
        return None

    parquet_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)

    print("\n--- CSV vs Parquet Benchmark ---")
    print(f"{'Metric':<20} {'CSV':>10} {'Parquet':>10}")
    print(f"{'Write time (s)':<20} {csv_write_time:>10.4f} {parquet_write_time:>10.4f}")
    print(f"{'Read time (s)':<20} {csv_read_time:>10.4f} {parquet_read_time:>10.4f}")
    print(f"{'File size (MB)':<20} {csv_size_mb:>10.3f} {parquet_size_mb:>10.3f}")

    os.remove(csv_path)
    return parquet_path


def verify_parquet(filepath):
    try:
        df_loaded = pd.read_parquet(filepath, engine="pyarrow")
    except Exception as e:
        print(f"Verify failed: {e}")
        return None

    print(f"\n--- Verify ---")
    print(f"Rows loaded back: {len(df_loaded)}")
    print(f"Cols loaded back: {list(df_loaded.columns)}")
    print(f"First timestamp:  {df_loaded['timestamp'].iloc[0]}")
    print(f"Last timestamp:   {df_loaded['timestamp'].iloc[-1]}")
    print(f"Any nulls:        {df_loaded.isnull().sum().sum()}")
    print(f"Dtype ofi:        {df_loaded.dtypes['ofi']}")
    return df_loaded


# ── TEST ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ticker = "SPY"
    df     = make_synthetic_features(n=100_000)

    print(f"DataFrame memory: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")

    filepath = benchmark_csv_vs_parquet(df, ticker)

    if filepath:
        df_back = verify_parquet(filepath)

        if df_back is not None:
            print("\n--- Expected vs Actual ---")
            print(f"Expected rows : 100000  | Actual: {len(df_back)}")
            print(f"Expected cols : 17      | Actual: {len(df_back.columns)}")
            print(f"Expected nulls: 0       | Actual: {df_back.isnull().sum().sum()}")
            print(f"Expected dtype: float32 | Actual: {df_back['ofi'].dtype}")
