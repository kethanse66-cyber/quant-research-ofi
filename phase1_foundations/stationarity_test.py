import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Download SPY data
data = yf.download("SPY", start="2020-01-01", end="2024-12-31")["Close"]["SPY"]

# Compute returns
returns = data.pct_change().dropna()

# ADF test on PRICE
result_price = adfuller(data)
print("ADF p-value for PRICE:", result_price[1])

# ADF test on RETURNS
result_returns = adfuller(returns)
print("ADF p-value for RETURNS:", result_returns[1])

# Rolling mean and std
rolling_mean = returns.rolling(20).mean()
rolling_std = returns.rolling(20).std()

# Plot
plt.figure(figsize=(10,4))
plt.plot(rolling_mean, label="Rolling Mean")
plt.plot(rolling_std, label="Rolling Std")
plt.legend()
plt.title("SPY Returns - Rolling Mean and Std")
plt.show()
