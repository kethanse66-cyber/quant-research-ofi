# quant-research-ofi
Cross-Asset Microstructure Alpha Signal | Intraday Liquidity Regimes & Order Flow Imbalance

## What This Project Is
A step-by-step quant research project targeting Jane Street, Citadel, and Two Sigma.
Building a Cross-Asset Microstructure Alpha Signal from scratch.

## Current Progress
- D1: Built return calculator using Python for loops
- D2: Rebuilt calculator using NumPy, learned indexing, log returns, variance, win rate
- D3: Loaded real SPY data, 8 DataFrame operations, 3 charts
- D4: Proved SPY returns are not normal. Skewness=-0.54, Kurtosis=11.44, P-value=0.0
- D5: Hypothesis test on SPY returns. Mean=0.00062, p-value=0.096, cannot confirm mean is real
- D6: SPY vs QQQ correlation=0.93. OLS regression beta=1.13, R-squared=0.87, P-value=0.0
- D7: ADF test on SPY. Price p-value=0.948 (not stationary). Returns p-value=0.0 (stationary)
- D8: Multiple testing on 20 random strategies. 1 fake signal before Bonferroni, 0 after correction
- pahse1-data pipline
- D1: Built simple order book. add_order, mid_price, calculate_ofi functions. Phase 1 started.
- D2: Built synthetic OFI pipeline. Generated bid/ask sizes, calculated OFI, spread, and rolling features
- D3: spread_calculator.py: Quoted spread, effective spread, Roll spread. Comparison chart plotted

## Files
### phase1_foundations/
- return_calculator.py: Daily returns, mean, variance, standard deviation using for loops
- return_calculator_numpy.py: Same calculator rewritten using NumPy, no for loops
- pandas_basics.py: Real SPY data, returns, price and returns charts
- stats_report.py: Proves SPY returns not normal. Skewness, kurtosis, normality test
- hypothesis_test.py: Tests if SPY mean daily return is statistically different from zero
- regression_analysis.py: SPY vs QQQ correlation and OLS regression, beta=1.13
- stationarity_test.py: ADF test on SPY price and returns. Rolling mean and std plotted
- multiple_testing_demo.py: Bonferroni correction on 20 random strategies


### phase1_synthetic_pipeline/
- simple_orderbook.py: Limit order book with add_order, mid_price, calculate_ofi functions
-  ofi_synthetic.py: Synthetic bid/ask data, OFI calculation, spread, rolling OFI features
- spread_calculator.py: Quoted spread, effective spread, Roll spread. Comparison chart plotted


## Skills Being Built
- Python, NumPy, Pandas
- Log Returns, Vectorised Operations
- Statistics, Market Microstructure
- Machine Learning

## Key Concepts Learned So Far
- Simple returns: price[1:] / price[:-1] - 1
- Log returns: np.log(price[1:] / price[:-1])
- DataFrame operations: shape, describe, isnull, loc, sort_values, rolling, resample, dropna
- Volatility clustering: big crashes and recoveries happen together
- COVID crash March 2020: SPY fell 9.9% in one day
- Null hypothesis: assumption we try to reject using data
- P-value: probability the result happened by random chance
- Correlation: measures direction of relationship between two assets
- OLS Regression: finds exact multiplier between two assets
- Beta: when SPY moves 1%, QQQ moves 1.13%
- R-squared: how much of QQQ movement is explained by SPY
- Stationarity: mean and variance do not change over time
- SPY price is NOT stationary (p=0.948) — it trends upward
- SPY returns ARE stationary (p=0.0) — they fluctuate around zero
- ADF test: p below 0.05 means stationary, p above 0.05 means not
- Multiple testing: testing 20 strategies always produces fake signals by luck
- Bonferroni correction: divide 0.05 by number of tests to get true threshold
- Limit order book: bids and asks stored as price-quantity dictionaries
- Mid price: (best_bid + best_ask) / 2
- OFI: sum(bid quantities) - sum(ask quantities). Positive = more buyers = price goes up
- - Order Flow Imbalance (OFI): delta_bid - delta_ask
- Positive OFI: buyers dominate
- Negative OFI: sellers dominate
- Spread: ask_price - bid_price
- Rolling window: smooths noisy microstructure signals
- Rolling mean: average OFI over window
- Rolling sum: cumulative order flow pressure
- Rolling std: volatility of order flow
- Synthetic order book: simulated bid/ask liquidity
- - Quoted spread: ask - bid, direct measure of trading cost
- Effective spread: 2 * abs(trade_price - mid_price), actual cost of trade
- Effective spread always smaller than quoted spread
- Roll spread: spread estimated from price autocorrelation, no bid/ask needed
- Negative price autocorrelation reveals bid-ask bounce pattern
