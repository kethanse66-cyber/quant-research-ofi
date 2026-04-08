import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Synthetic Data
np.random.seed(42)
bid = 100 + np.cumsum(np.random.randn(100) * 0.01)
ask = bid + np.random.uniform(0.01, 0.05, size=100)
df = pd.DataFrame({'bid': bid, 'ask': ask})

# Quoted Spread
df['quoted_spread'] = df['ask'] - df['bid']

# Mid Price
df['mid_price'] = (df['bid'] + df['ask']) / 2

# Effective Spread
df['trade_price'] = df['bid'] + np.random.uniform(0, 1, size=100) * df['quoted_spread']
df['effective_spread'] = 2 * abs(df['trade_price'] - df['mid_price'])

# Roll Spread
df['price_change'] = df['mid_price'].diff()
roll_cov = df['price_change'].cov(df['price_change'].shift(1))
roll_spread = 2 * np.sqrt(-roll_cov)

# Print Results
print(df.head(10))
print("Average Quoted Spread:", df['quoted_spread'].mean())
print("Average Effective Spread:", df['effective_spread'].mean())
print("Roll Spread:", roll_spread)

# Plot
plt.plot(df['quoted_spread'], label='quoted_spread')
plt.plot(df['effective_spread'], label='effective_spread')
plt.xlabel("Time")
plt.ylabel("Spread Value")
plt.title("Quoted vs Effective Spread")
plt.legend()
plt.show()
