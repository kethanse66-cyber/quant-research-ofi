# spread_calculator.py
# Three ways to measure the bid-ask spread.
# Reference: Roll (1984)
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
np.random.seed(42)
N = 200
 
 
def generate_quotes(n):
    """Fake bid and ask prices with a random-walk mid."""
    bid = 100 + np.cumsum(np.random.randn(n) * 0.01)
    ask = bid + np.random.uniform(0.01, 0.05, size=n)
    return pd.DataFrame({"bid": bid, "ask": ask})
 
 
def quoted_spread(df):
    """What you see on screen: ask minus bid.
 
    This is the maximum you pay if you trade at the posted quotes
    with no price improvement.
    """
    return df["ask"] - df["bid"]
 
 
def mid_price(df):
    """Simple average of bid and ask."""
    return (df["bid"] + df["ask"]) / 2
 
 
def simulate_trades(df):
    """Assign each trade a direction: buy hits ask, sell hits bid.
 
    Direction is required for effective spread. Without it you cannot
    measure how much of the spread was actually paid.
    """
    direction = np.where(np.random.rand(len(df)) > 0.5, 1, -1)
    price = np.where(direction == 1, df["ask"].values, df["bid"].values)
    return (pd.Series(price, index=df.index),
            pd.Series(direction, index=df.index))
 
 
def effective_spread(trade_price, direction, mid):
    """Actual round-trip cost paid — may be less than quoted spread.
 
    Formula: 2 * direction * (trade_price - mid_price)
    Factor of 2 makes it comparable to quoted spread (round-trip).
    Effective spread < quoted spread when price improvement happens.
    """
    return 2 * direction * (trade_price - mid)
 
 
def roll_spread(mid):
    """Implied spread from return autocorrelation — Roll (1984).
 
    Bid-ask bounce causes negative serial correlation in price changes.
    This lets you back out the spread without needing quote data.
    Formula: 2 * sqrt(-cov(delta_p_t, delta_p_{t-1}))
 
    Returns NaN if covariance is positive — model assumption violated.
    """
    delta_p = mid.diff()
    cov = delta_p.cov(delta_p.shift(1))
    if cov >= 0:
        print(f"WARN: Roll covariance is {cov:.6f} (positive). Returning NaN.")
        return np.nan
    return 2 * np.sqrt(-cov)
 
 
if __name__ == "__main__":
    df = generate_quotes(N)
    df["quoted_spread"]    = quoted_spread(df)
    df["mid"]              = mid_price(df)
    df["trade_price"], df["direction"] = simulate_trades(df)
    df["effective_spread"] = effective_spread(
        df["trade_price"], df["direction"], df["mid"]
    )
    rs = roll_spread(df["mid"])
 
    print("=== Spread Summary ===")
    print(f"Mean quoted spread    : {df['quoted_spread'].mean():.5f}")
    print(f"Mean effective spread : {df['effective_spread'].mean():.5f}")
    if not np.isnan(rs):
        print(f"Roll implied spread   : {rs:.5f}")
    else:
        print("Roll spread           : NaN")
 
    print("\nFirst 10 rows:")
    print(df[["bid", "ask", "quoted_spread", "effective_spread"]].head(10))
 
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["quoted_spread"].values, label="quoted spread", linewidth=1)
    ax.plot(df["effective_spread"].values, label="effective spread",
            linewidth=1, alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Spread value")
    ax.set_title("Quoted vs effective spread")
    ax.legend()
    plt.tight_layout()
    plt.savefig("/tmp/spread_comparison.png", dpi=120)
    print("\nPlot saved to /tmp/spread_comparison.png")
 
