# quant-research-ofi
Cross-Asset Microstructure Alpha Signal | Intraday Liquidity Regimes & Order Flow Imbalance

## What This Project Is
An institutional-grade quant research project targeting Jane Street, Citadel, and Two Sigma.
Building a Cross-Asset OFI Alpha Signal from scratch — data pipeline, regime detection, predictive model, backtesting engine.

## Current Progress

### Phase 0 — Foundations
- D1: Return calculator using Python for loops
- D2: Rebuilt using NumPy. Log returns, win rate, vectorised operations
- D3: Real SPY data. 8 DataFrame operations. 3 charts
- D4: Proved SPY returns not normal. Skewness=-0.54, Kurtosis=11.44, P-value=0.0
- D5: Hypothesis test on SPY mean return. t-stat=1.66, p-value=0.096
- D6: SPY vs QQQ correlation=0.93. OLS beta=1.13, R-squared=0.87
- D7: ADF test. SPY price not stationary p=0.948. Returns stationary p=0.0
- D8: Multiple testing on 20 strategies. 1 fake signal before Bonferroni, 0 after

### Phase 1 — Synthetic Pipeline
- D1 (Apr 6): Simple order book. add_order, mid_price, calculate_ofi functions
- D2 (Apr 7): Synthetic OFI pipeline. Bid/ask sizes, OFI, spread, rolling features
- D3 (Apr 8): Spread calculator. Quoted spread, effective spread, Roll spread. Comparison chart
- D4 (Apr 9): Tick cleaner. Drop duplicates, remove bad prices, drop zero volume, fill missing timestamps, UTC normalization
- D5 (Apr 10): Volume bar builder. Volume bars vs time bars comparison.Time bar volume std=8446 vs mean=28951. Volume bars normalize activity per bar.Quote data OFI input. Trade data = price and volume input.

## Files

### phase0_foundations/
- return_calculator.py
- return_calculator_numpy.py
- pandas_basics.py
- stats_report.py
- hypothesis_test.py
- regression_analysis.py
- stationarity_test.py
- multiple_testing_demo.py

### phase1_synthetic_pipeline/
- simple_orderbook.py
- ofi_synthetic.py
- spread_calculator.py
- tick_cleaner.py
- volume_bar_builder.py

## Key Concepts
- OFI: delta_bid - delta_ask. Positive = buy pressure. Negative = sell pressure
- Quoted spread: ask - bid. Direct cost of trading
- Effective spread: 2 * abs(trade_price - mid_price). Actual trade cost
- Roll spread: estimated from price autocorrelation alone. No bid/ask needed
- Stationarity: required before any regression. ADF test to verify
- Multiple testing: always apply Bonferroni correction or results are meaningless
- Beta: sensitivity of one asset to another. SPY vs QQQ beta = 1.13
- Volume bars: one bar per X shares traded. Equal activity per bar.Better than time bars for microstructure research
- Trade data: actual transactions. Price and volume per tick
- Quote data: bid/ask offers. bid_size and ask_size changes → OFI input
- OHLC: open high low close. Built from compressing ticks. Not used in OFI pipeline

