# D3 — Pandas Basics + DataFrame Operations
# quant-research-ofi | phase1_foundations

# Import libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download 5 years of SPY data
df = yf.download("SPY", start='2020-01-01', end='2024-12-31')

# Fix double header from yfinance
df.columns = df.columns.get_level_values(0)

# Calculate daily returns
df['returns'] = df['Close'].pct_change()

# Check size of data
print(df.shape)

# Statistics of all columns
print(df.describe())

# Count missing values
print(df.isnull().sum())

# COVID crash period
print(df.loc['2020-03-01':'2020-03-31'])

# 5 worst days
print(df.sort_values('returns').head())

# 5 best days
print(df.sort_values('returns', ascending=False).head())

# 20 day rolling average
df['rolling_mean'] = df['returns'].rolling(20).mean()

# Plot returns vs rolling average
plt.plot(df['returns'], label='Daily Returns')
plt.plot(df['rolling_mean'], label='20 Day Average', color='red')
plt.title("SPY Returns vs 20 Day Rolling Average")
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.show()

# Monthly returns
print(df['returns'].resample("ME").sum())

# Remove missing values
df_clean = df.dropna()
print(df_clean.shape)

# Save clean data
df.to_csv("spy_data_clean.csv")
```

---

