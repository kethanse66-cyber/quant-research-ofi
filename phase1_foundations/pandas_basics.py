!pip install yfinance
import matplotlib.pyplot as plt
import yfinance as yf

df = yf.download("SPY", start="2020-01-01", end="2024-12-31")
print(df)

df.columns = df.columns.get_level_values(0)

df['returns'] = df['Close'].pct_change()
print(df['returns'])

plt.plot(df['returns'])
plt.title("Daily Returns Data")
plt.xlabel('Date')
plt.ylabel('Daily Returns')
plt.show()

df.to_csv("spy_data.csv")
# D3 — Pandas Basics + DataFrame Operations
# quant-research-ofi | phase1_foundations

# ── IMPORTS ──────────────────────────────
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ── DOWNLOAD DATA ─────────────────────────
df = yf.download("SPY", start="2020-01-01", end="2024-12-31")
df.columns = df.columns.get_level_values(0)

# ── ADD RETURNS ───────────────────────────
df['returns'] = df['Close'].pct_change()

# ── OPERATION 1 — SHAPE ───────────────────
print("Shape:")
print(df.shape)

# ── OPERATION 2 — DESCRIBE ────────────────
print("\nDescribe:")
print(df.describe())

# ── OPERATION 3 — MISSING VALUES ──────────
print("\nMissing Values:")
print(df.isnull().sum())

# ── OPERATION 4 — SELECT BY DATE ──────────
print("\nCOVID Crash March 2020:")
crash = df.loc['2020-03-01':'2020-03-31']
print(crash)

# ── OPERATION 5 — SORT VALUES ─────────────
print("\n5 Worst Days:")
worst_days = df.sort_values('returns')
print(worst_days.head())

print("\n5 Best Days:")
best_days = df.sort_values('returns', ascending=False)
print(best_days.head())

# ── OPERATION 6 — ROLLING AVERAGE ─────────
df['rolling_days'] = df['returns'].rolling(20).mean()
print("\nRolling Average:")
print(df[['returns', 'rolling_days']].head(25))

# ── OPERATION 7 — RESAMPLE MONTHLY ────────
print("\nMonthly Returns:")
monthly_returns = df['returns'].resample('ME').sum()
print(monthly_returns)

# ── OPERATION 8 — DROPNA ──────────────────
df_clean = df.dropna()
print("\nShape after dropna:")
print(df_clean.shape)

# ── PLOT 1 — CLOSING PRICE ────────────────
plt.plot(df['Close'])
plt.title('SPY Closing Price 2020 to 2024')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# ── PLOT 2 — DAILY RETURNS ────────────────
plt.plot(df['returns'])
plt.title('SPY Daily Returns')
plt.xlabel('Date')
plt.ylabel('Daily Returns')
plt.show()

# ── PLOT 3 — RETURNS VS ROLLING AVERAGE ───
plt.plot(df['returns'], label='Daily Returns')
plt.plot(df['rolling_days'], label='20 Day Average', color='red')
plt.title('SPY Returns vs 20 Day Rolling Average')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.show()

# ── SAVE CLEAN DATA ───────────────────────
df.to_csv('spy_data_clean.csv')
print("\nData saved to spy_data_clean.csv")
