# quant-research-ofi
 Cross-Asset Microstructure Alpha Signal |                Intraday Liquidity Regimes &amp; Order Flow Imbalance
# Quant Research Project

## What This Project Is
A step-by-step quant research project targeting Jane Street, Citadel, and Two Sigma.
Building a Cross-Asset Microstructure Alpha Signal from scratch.

## Current Progress
- D1: Built return calculator using Python for loops
- D2: Rebuilt calculator using NumPy, learned indexing,log returns, variance, win rate
- D3: Loaded real SPY data, 8 DataFrame operations, 3 charts
- D4: Proved SPY returns are not normal. Skewness=-0.54, Kurtosis=11.44, P-value=0.0
- D5: Hypothesis test on SPY returns.  Mean = 0.00062, p-value = 0.096, cannot confirm mean is real
- D6: SPY vs QQQ correlation=0.93. OLS regression beta=1.13, R-squared=0.87, P-value=0.0
- D7: ADF test on SPY. Price p-value=0.948 (not stationary).  Returns p-value=0.0 (stationary). Rolling mean and std plotted.
- D8: Multiple testing on 20 random strategies. 1 fake signal  before Bonferroni, 0 after correction. Threshold dropped to 0.0025.



## Files
- return_calculator.py: Calculates daily returns, mean, variance, and standard deviation
- return_calculator_numpy.py: Same calculator rewritten using NumPy, no for loops
- pandas_basics.py: Loads real SPY data, calculates returns, plots price and returns charts
- stats_report.py: Proves SPY returns are not normal using skewness, kurtosis, normality test
- hypothesis_test.py: Tests if SPY mean daily return is - statistically different from zero using ttest_1samp
- regression_analysis.py: SPY vs QQQ correlation and OLS regression, beta=1.13, R-squared=0.87
-  stationarity_test.py: ADF stationarity test on SPY price and returns. rolling 20-day mean and std plotted.
-  multiple_testing_demo.py: Bonferroni correction on 20 random  trategies. Shows how fake signals disappear after correction.
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
- Null hypothesis: assumption we try to reject using data
- P-value: probability the result happened by random chance
- Correlation: measures direction of relationship between two assets
- OLS Regression: finds exact multiplier between two assets
- Beta: when SPY moves 1%, QQQ moves 1.13%
- R-squared: how much of QQQ movement is explained by SPY
-  Stationarity: a series is stationary if mean and variance dont change over time
- SPY price is NOT stationary (p=0.948) — it trends upward
- SPY returns ARE stationary (p=0.0) — they fluctuate around zero
- ADF test: p below 0.05 means stationary, p above 0.05 means not stationary
- Always test stationarity before running any regression
- - Multiple testing: testing 20 strategies always produces fake signals by luck
- Bonferroni correction: divide 0.05 by number of tests to get true threshold
- 20 tests → threshold becomes 0.05/20 = 0.0025
- Before correction: 1 strategy looked real. After: 0 survived.
- This is why most backtests are lies
