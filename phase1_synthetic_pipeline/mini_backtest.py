import yfinance as yf
import pandas as pd

# download
df = yf.download("SPY", interval="1m", period="1d")
df = df.dropna()

# features
df["returns"] = df["Close"].pct_change()
df["rolling_mean"] = df["returns"].rolling(20).mean()
df["lag_returns"] = df["returns"].shift(1)

# target (future return)
df["target"] = df["returns"].shift(-1)

# remove warmup rows
df = df.dropna()

# signal
df["signal"] = df["lag_returns"] > df["rolling_mean"]

# position
df["position"] = df["signal"].astype(int)

# strategy return
df["strategy"] = df["position"] * df["target"]

# equity curve
df["equity"] = (1 + df["strategy"]).cumprod()

print(df.head())
print(df.tail())
