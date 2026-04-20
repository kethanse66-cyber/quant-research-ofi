# Progress Log

## Phase 0 — Foundations

### D1 — Return Calculator
**Learned:** Python for loops, lists, basic math operations
**Built:** return_calculator.py — daily returns, mean, variance, std using for loops
**Confused:** Nothing major

---

### D2 — NumPy Foundations
**Learned:** NumPy arrays, vectorised returns, log returns, indexing, conditional filtering, win rate, expected value, basic probability
**Built:** return_calculator_numpy.py — max, min, mean, std, win rate, log returns, no for loops
**Confused:** Log returns math, e and natural log. Mental math multiplication still slow

---

### D3 — Pandas Basics
**Learned:** DataFrame operations, pct_change, rolling mean, resample, isnull, loc, sort_values, dropna. Read COVID crash March 2020 in real data
**Built:** pandas_basics.py — SPY closing price chart, daily returns chart, rolling average chart, 5 worst and best days
**Confused:** Nothing major

---

### D4 — Descriptive Stats + SciPy
**Learned:** Skewness, kurtosis, normality test, histogram with normal curve overlay, density=True, np.linspace, stats.norm.pdf
**Built:** stats_report.py — proved SPY returns not normal three ways. Skewness=-0.54, Kurtosis=11.44, P-value=0.0
**Confused:** Nothing major

---

### D5 — Hypothesis Testing
**Learned:** Null hypothesis, t-statistic, p-value, ttest_1samp, f-strings, round()
**Built:** hypothesis_test.py — tested if SPY mean return differs from zero. t-stat=1.66, p-value=0.096, cannot confirm
**Confused:** Nothing major

---

### D6 — Correlation + OLS Regression
**Learned:** Correlation direction, SPY vs QQQ=0.93, OLS beta, R-squared, sm.add_constant, residual skew and kurtosis
**Built:** regression_analysis.py — SPY vs QQQ correlation, OLS regression, scatter plot. Beta=1.13, R-squared=0.87
**Confused:** Nothing major

---

### D7 — Time Series + Stationarity
**Learned:** Stationarity, ADF test, SPY price NOT stationary p=0.948, SPY returns stationary p=0.0, rolling mean and std
**Built:** stationarity_test.py — ADF test on SPY price and returns, rolling mean and std chart
**Confused:** Nothing major

---

### D8 — Multiple Testing Problem
**Learned:** Why many strategies produce fake signals, Bonferroni correction, new threshold = 0.05 / number of tests
**Built:** multiple_testing_demo.py — 20 random strategies, Bonferroni applied, 1 fake signal before, 0 after
**Confused:** Nothing major

---

## Phase 1 — Synthetic Pipeline

### D1 (Apr 6) — Simple Order Book
**Learned:** Limit order book structure, bid and ask dictionaries, bid-ask spread, mid price formula, OFI formula
**Built:** simple_orderbook.py — add_order, mid_price, calculate_ofi functions
**Confused:** Nothing major
### D2 (Apr 7) — Synthetic OFI Pipeline
**Learned:** OFI using delta bid and delta ask, synthetic order book generation, rolling window features, OFI positive=buy pressure, OFI negative=sell pressure
**Built:** ofi_synthetic.py — synthetic bid/ask sizes, OFI calculation, spread, rolling OFI mean, sum, std
**Confused:** np.convolve vs pandas rolling. Rolling window meaning in ticks vs days
### D3 (Apr 8) — Spread Calculator
**Learned:** Quoted spread, effective spread, Roll spread, negative autocorrelation reveals bid-ask bounce, mid price
**Built:** spread_calculator.py — quoted spread, effective spread, Roll spread, comparison chart
**Confused:** Nothing major
### D4 (Apr 9) — Tick Cleaner
**Learned:** Tick data cleaning pipeline, duplicate timestamps, missing price handling, bad price filtering, zero volume filtering, timestamp normalization, UTC **conversion,** resampling to fill missing timestamps
**Built:** tick_cleaner.py — synthetic tick data, duplicate injection, missing value handling, bad price removal, drop zero volume, resample to 1-second grid, UTC normalization, parquet save
Confused: Difference between tz_localize vs tz_convert, when to resample vs forward fill
### D5 (Apr 10) — Volume Bar Builder
**Learned:** Time bars vs volume bars difference, OHLC construction from ticks,
volume bars normalize market activity, quote data vs trade data difference,
Polygon gives two separate files — quotes for OFI, trades for price and volume,
### D6 (Apr 11) — Saturday Review
**Learned:** Review day — no new file
**Built:** Nothing new — review and GitHub push day
**Confused:** Nothing major

### D7 (Apr 13) — Trade Imbalance
**Learned:** Lee-Ready rule to classify trades as buy or sell, volume-weighted trade imbalance, rolling window imbalance, why we use volume not just count, direction +1/-1/0 meaning
**Built:** trade_imbalance.py — classify_trade function, trade_imbalance function, rolling buy/sell volume, imbalance formula (buys-sells)/(buys+sells)
**Confused:** nested np.where syntax, why direction=0 at mid price
bid_size and ask_size are orders waiting in book not volume traded
**Built:** volume_bar_builder.py — volume bars with VWAP and total volume,
time bars with OHLC for comparison, volume stats comparison
**Confused:** Nothing major

### D8 (Apr 14): Feature Library
**Learned:** Microprice weighted by bid/ask size, spread change as liquidity signal, OFI normalization using rolling standard deviation, why normalization stabilizes scale across time  
**Built:** feature_library.py — compute_ofi function, microprice, spread change, normalized OFI features  
**Confused:** Why normalize OFI and not use raw OFI

### D9 (Apr 15): Queue Imbalance
**Learned:** Queue imbalance at best level, why Level 1 only matters for Polygon data, edge case handling with np.where
**Built:** queue_imbalance.py — queue_imbalance_best function, edge case tests for bid=0 ask=0
**Confused:** Nothing major

### D10 (Apr 16): OFI Full Analysis  
**Learned:** Multi-horizon OFI at 30s, 1min, 5min bars, autocorrelation, ADF stationarity test, Information Coefficient (IC), lagged OFI avoids look-ahead bias  
**Built:** ofi_full.py — tick-level OFI, horizon resampling, ACF analysis, ADF test, IC calculation, visualization dashboard  
**Confused:** Why shorter or longer time horizons give different IC results

### D11 (Apr 17): Feature Normalizer
**Learned:** Three normalization methods — rank transform, z-score, min-max. Rolling window versions avoid look-ahead bias. Rank transform preferred before HMM because OFI is fat-tailed and skewed — rank removes the distributional shape entirely. Z-score and min-max preserve the original skew.
**Built:** feature_normalizer.py — zscore_normalize, minmax_normalize, rank_transform (full sample + rolling), normalize_features wrapper for full DataFrame, normalization_audit for before/after stats
**Confused:** Nothing major

### D13 (Apr 20): Target Variable
**Learned:** Log returns at 3 horizons — 10s, 1min, 5min. shift(-n) works in rows not seconds. 
1 row = 10 seconds so 1min = shift(-6), 5min = shift(-30). 
Last N rows get NaN because no future rows exist after them. 
Index frequency check confirms all rows are exactly 10s apart — critical for correct shift numbers.
Log returns used instead of raw price difference because comparable across all tickers.
**Built:** target_variable.py — compute_log_returns function, parameterized horizons dictionary, index frequency validation, NaN count test
**Confused:** NaN concept took time — understood finally through apple box example


