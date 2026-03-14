import numpy as np
prices = np.array([100,102,101,105,103])
print(prices)
returns = prices[1:]/prices[:-1]-1
print("Std deviation:",round(np.std(returns)*100,2),"%")
print("best day: ",round(np.max(returns)*100,2),"%")
print("worst day: ",round(np.min(returns)*100,2),"%")
print("average: ",round(np.mean(returns)*100,2),"%")
