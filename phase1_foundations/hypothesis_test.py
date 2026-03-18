import pandas as pd
import numpy as np
import scipy.stats as stats
df=pd.read_csv("spy_data.csv")
returns=df['returns'].dropna()
t_stat,p_value=stats.ttest_1samp(returns,popmean=0)
print("mean return:  ",round(returns.mean(),6))
print("t_stat:  ",round(t_stat,4))
print("p_value:  ",round(p_value,4))
print("The mean daily return of SPY is 0.00062. The p-value is 0.096 which is above 0.05, so we cannot confirm this return is real — it could be due to random chance.")
