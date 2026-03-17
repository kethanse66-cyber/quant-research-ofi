import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
df=pd.read_csv("spy_data.csv",index_col='Date',parse_dates=True)
returns=df['returns'].dropna()
print(stats.describe(returns))

plt.hist(returns,bins=50,density=True,color='blue',alpha=0.6)
xmin,xmax=returns.min(),returns.max()
x=np.linspace(xmin,xmax,100)
mu,std=returns.mean(),returns.std()
normal_curve=stats.norm.pdf(x,mu,std)
plt.plot(x,normal_curve,color='red',linewidth=2)
plt.title('SPY RETURNS VS NORMAL DISTRIBUTION')
plt.xlabel("returns")
plt.ylabel("density")
plt.show()
stat,p=stats.normaltest(returns)
print("statistic:  ",round(stat,4))
print("p:  ",round(p,6))
# SPY returns are not normally distributed. P-value = 0.0 confirms this.
# Skewness = -0.54 means SPY crashes harder than it rallies.
# Kurtosis = 11.44 means extreme days happen far more than normal predicts.

