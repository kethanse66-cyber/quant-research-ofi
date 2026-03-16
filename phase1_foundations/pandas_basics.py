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
