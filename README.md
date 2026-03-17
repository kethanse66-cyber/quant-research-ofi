# quant-research-ofi
 Cross-Asset Microstructure Alpha Signal |                Intraday Liquidity Regimes &amp; Order Flow Imbalance
# Quant Research Project

## What This Project Is
A step-by-step quant research project targeting Jane Street, Citadel, and Two Sigma.
Building a Cross-Asset Microstructure Alpha Signal from scratch.

## Current Progress
- D1: Built return calculator using Python for loops
- D2: Rebuilt calculator using NumPy, learned indexing,log returns, variance, win rate
-  D3: Loaded real SPY data, 8 DataFrame operations, 3 charts
-  D4: Proved SPY returns are not normal. Skewness=-0.54, Kurtosis=11.44, P-value=0.0



## Files
- return_calculator.py: Calculates daily returns, mean, variance, and standard deviation
- return_calculator_numpy.py: Same calculator rewritten using NumPy, no for loops
- pandas_basics.py: Loads real SPY data, calculates returns, plots price and returns charts
- stats_report.py: Proves SPY returns are not normal using skewness, kurtosis, normality test

## Skills Being Built
- Python
- numpy
- pandas
- Log Returns
- Vectorised Operations
- Statistics
- Market Microstructure
- Machine Learning

 
- Key Concepts Learned So Far
- Simple returns: price[1:] / price[:-1] - 1
- Log returns: np.log(price[1:] / price[:-1])
- DataFrame operations: shape, describe, isnull, loc, 
  sort_values, rolling, resample, dropna
- Volatility clustering: big crashes and recoveries happen together
- COVID crash March 2020: SPY fell 9.9% in one month
