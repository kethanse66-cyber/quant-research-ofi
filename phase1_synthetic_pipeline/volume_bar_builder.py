# volume_bar_builder.py
# Builds volume bars from tick data and compares them to time bars.
# Reference: Lopez de Prado (2018) — Advances in Financial Machine Learning
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
VOLUME_THRESHOLD = 500
 
 
def assign_bar_ids(volume_series, threshold):
    """Give each tick a bar ID based on cumulative volume.
 
    A new bar starts every time cumulative volume crosses the threshold.
    If one tick is bigger than the threshold, the leftover carries forward
    into the next bar — this is the spillover fix from Lopez de Prado (2018).
 
    Without the spillover fix, a 800-share tick with threshold=500 would
    just close bar 0 at 800 shares. With the fix, bar 0 closes at 500
    and bar 1 starts with 300 already accumulated. Much more accurate.
    """
    bar_ids = np.zeros(len(volume_series), dtype=int)
    cumvol = 0
    bar_id = 0
 
    for i, vol in enumerate(volume_series):
        cumvol += vol
        while cumvol >= threshold:
            cumvol -= threshold
            bar_id += 1
        bar_ids[i] = bar_id
 
    return bar_ids
 
 
def build_volume_bars(df, threshold):
    """Aggregate ticks into OHLCV bars where each bar = N shares traded.
 
    Volume bars have more uniform information content than time bars.
    During busy periods (market open) bars form quickly.
    During quiet periods they form slowly.
    This matters for OFI — each bar represents the same market activity.
 
    Uses groupby.agg() not groupby.apply() — much faster on large datasets.
    """
    df = df.copy()
    df["bar_id"] = assign_bar_ids(df["volume"], threshold)
 
    bars = df.groupby("bar_id").agg(
        timestamp    = ("timestamp", "last"),
        open         = ("price",     "first"),
        high         = ("price",     "max"),
        low          = ("price",     "min"),
        close        = ("price",     "last"),
        total_volume = ("volume",    "sum"),
    )
 
    bars["vwap"] = (
        (df["price"] * df["volume"])
        .groupby(df["bar_id"])
        .sum() / bars["total_volume"]
    )
 
    return bars.reset_index(drop=True)
 
 
def build_time_bars(df, freq="1min"):
    """Aggregate ticks into fixed-time OHLCV bars — for comparison only.
 
    Time bars are not used in the main pipeline. They are here to show
    that volume bars are more uniform in information content.
    Empty bars (quiet periods with no trades) are dropped.
    """
    df = df.copy()
    ts = df["timestamp"]
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        df["timestamp"] = ts.dt.tz_localize(None)
 
    df = df.set_index("timestamp")
    ohlc = df["price"].resample(freq).ohlc()
    ohlc["volume"] = df["volume"].resample(freq).sum()
    ohlc = ohlc.dropna()
    return ohlc.reset_index()
 
 
def compare_bar_uniformity(vol_bars, time_bars):
    """Print volume stats for both bar types side by side.
 
    Lower CV (std/mean) for volume bars confirms more uniform information.
    """
    vb = vol_bars["total_volume"]
    tb = time_bars["volume"].dropna()
    print(f"{'Metric':<25} {'Volume bars':>14} {'Time bars':>14}")
    print("-" * 55)
    print(f"{'Mean volume':<25} {vb.mean():>14.1f} {tb.mean():>14.1f}")
    print(f"{'Std volume':<25} {vb.std():>14.1f}  {tb.std():>14.1f}")
    print(f"{'CV (std/mean)':<25} {vb.std()/vb.mean():>14.4f} {tb.std()/tb.mean():>14.4f}")
    print(f"{'Num bars':<25} {len(vb):>14d} {len(tb):>14d}")
    print("\nLower CV = more uniform information content per bar.")
 
 
def verify_spillover_fix():
    """Check that a single large tick splits correctly across bars.
 
    With threshold=500 and tick volume=800:
    - Naive: bar 0 gets 800 shares (wrong)
    - Correct: bar 0 gets 500, bar 1 starts with 300
    """
    test_vol = pd.Series([200, 800, 100, 400, 600])
    ids = assign_bar_ids(test_vol, threshold=500)
    expected = np.array([0, 2, 2, 3, 4])
 
    if np.array_equal(ids, expected):
        print(f"PASS — spillover fix correct: {ids}")
    else:
        print(f"FAIL — got {ids}, expected {expected}")
 
 
if __name__ == "__main__":
    verify_spillover_fix()
    print()
 
    df = pd.read_parquet("clean_ticks.parquet")
    print(f"Loaded {len(df)} clean ticks\n")
 
    vol_bars  = build_volume_bars(df, VOLUME_THRESHOLD)
    time_bars = build_time_bars(df, freq="1min")
 
    print("=== Volume Bar Sample (first 8) ===")
    print(vol_bars.head(8).to_string(index=False))
 
    print("\n=== Uniformity Comparison ===")
    compare_bar_uniformity(vol_bars, time_bars)
 
    vol_bars.to_parquet("volume_bars.parquet", index=False)
    print("\nvolume_bars.parquet saved")
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(vol_bars["total_volume"], bins=20, edgecolor="white")
    axes[0].set_title("Volume per bar — volume bars")
    axes[0].set_xlabel("Total volume")
    axes[1].hist(time_bars["volume"].dropna(), bins=20, edgecolor="white")
    axes[1].set_title("Volume per bar — time bars")
    axes[1].set_xlabel("Total volume")
    plt.tight_layout()
    plt.savefig("/tmp/bar_comparison.png", dpi=120)
    print("Plot saved to /tmp/bar_comparison.png")
