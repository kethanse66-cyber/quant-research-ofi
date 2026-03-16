# D3 Bonus — Crash Analyser
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

# Find crash days — returns worse than -2%
crash_days = df[df['returns'] < -0.02]
print("Total crash days:", len(crash_days))

# Find rally days — returns better than +2%
rally_days = df[df['returns'] > 0.02]
print("Total rally days:", len(rally_days))

# Create colour for each day
colors = ['red' if r < -0.02 else 'green' if r > 0.02 else 'blue'
          for r in df['returns']]

# Plot colour coded returns chart
plt.figure(figsize=(15, 5))
plt.bar(df.index, df['returns'], color=colors, width=1)
plt.title('SPY Daily Returns — Crash vs Rally Days')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.axhline(y=0, color='black', linewidth=1.5)
plt.show()

# Find worst and best month
monthly = df['returns'].resample('ME').sum()
print("Worst month:", monthly.idxmin(), monthly.min())
print("Best month:", monthly.idxmax(), monthly.max())
```

---


