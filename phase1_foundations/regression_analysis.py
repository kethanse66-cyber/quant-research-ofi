import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
data=yf.download(['SPY','QQQ'],start='2020-01-01',end='2024-12-31')['Close']
print(data.head())
returns=data.pct_change().dropna()
print(returns.head())
corelation=returns['SPY'].corr(returns['QQQ'])
print(corelation)
x=returns['SPY']
y=returns['QQQ']
x=sm.add_constant(x)
print(x.head())
model=sm.OLS(y,x).fit()
print(model.summary())
print("BETA: ",model.params["SPY"])
print("r squared: ",model.rsquared)
print("p_value: ",model.pvalues["SPY"])
plt.scatter(returns["SPY"],returns["QQQ"],alpha=0.3)
plt.xlabel('SPY')
plt.ylabel("QQQ")
plt.title("SPY VS QQQ daily returns")
plt.show()
