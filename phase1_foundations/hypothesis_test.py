import pandas as pd
import numpy as np
import scipy.stats as stats
df=pd.read_csv("spy_data.csv")
returns=df['returns'].dropna()
t_stat,p_value=stats.ttest_1samp(returns,popmean=0)
print("mean return:  ",round(returns.mean(),6))
print("t_stat:  ",round(t_stat,4))
print("p_value:  ",round(p_value,4))
if p_value<0.05:
  print("Statistically significant result (reject H0)")
else:
  print("Not statistically significant (fail to reject H0)")
