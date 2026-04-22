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
- D5 (Apr 10): Volume bar builder. Volume bars vs time bars comparison.Time bar volume std=8446 vs mean=28951. Volume bars normalize activity per bar.Quote data      OFI input. Trade data = price and volume input.
- D6 (Apr 11): Saturday review day. No new file. GitHub push and cleanup
- D7 (Apr 13): Trade imbalance. Lee-Ready classification. Volume-weighted rolling imbalance. Range -1 to +1
- D8 (Apr 14): Feature library. Microprice, spread change, normalized OFI. Weighted mid-price using queue sizes. OFI normalized using rolling std.
- D9 (Apr 15): Queue imbalance. Best level formula. Edge case tests. Level 1 only for Polygon data.
- D10 (Apr 16): Multi-horizon OFI (30s / 1min / 5min), ACF, ADF stationarity, Information Coefficient, visualization dashboard
- D11 (Apr 17): Feature normalizer. Rank transform, z-score, min-max with rolling window support. Rank transform chosen for HMM — removes fat-tail shape. Audit function shows skew and kurtosis before vs after.
-  D13 (Apr 20): Target variable. Log returns at 10s, 1min, 5min horizons. Parameterized horizons dict. Index frequency validation. NaN count confirmed correct.
- D14 (Apr 21): Lag features. shift(1,2,3) on all features. Rolling normalization with shift(1) to avoid look-ahead bias.
- D15 (Apr 22): Audit pipeline. Look-ahead audit on all features. Lag validation confirmed — df_model.iloc[0] matches df_raw.iloc[0]. All features PASS. 2 rows dropped — first row NaN features, last row NaN target. First valid prediction row 09:31am.
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
- trade_imbalance.py
- feature_library.py
- queue_imbalance.py
- ofi_full.py
- feature_normalizer.py
- target_variable.py
- lag_features.py
- audit_pipeline.py

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
-Trade imbalance: (buy_vol - sell_vol) / (buy_vol + sell_vol). Range -1 to +1. Measures directional volume pressure
- Lee-Ready rule: trade above mid = buy (+1). Trade below mid = sell (-1). Trade at mid = neutral (0)
- Microprice: weighted mid price using queue sizes. Closer to side with larger size
- Spread change: change in ask-bid spread. Measures liquidity tightening/widening
- Normalized OFI: OFI divided by rolling std. Measures imbalance strength relative to recent history
- Queue imbalance: (bid_size - ask_size) / (bid_size + ask_size). Range -1 to +1. Measures directional pressure at best level
- Multi-horizon OFI: OFI tested at 30s, 1min, 5min bars
- ACF: checks if OFI continues or reverses over time
- ADF test: checks if OFI is stationary
- IC: correlation between OFI and future returns
- Lagged OFI: previous OFI used to avoid look-ahead bias
- - Target variable: log(price_t+n / price_t). What price actually did n seconds later. Model predicts this using OFI
- Log returns: log(future/current). Stationary and comparable across tickers. Raw price difference is neither
- shift(-n): looks n rows ahead. Rows not seconds. 1min = shift(-6) only if each row = 10 seconds
- Index frequency check: confirms row spacing before trusting shift numbers. Critical for real Polygon data
- NaN in targets: last N rows have no future data. Dropped before model training. Not an error
- Rank transform: convert values to percentile ranks 0-1. Removes distributional shape. Preferred before HMM fitting
- Z-score: (x - mean) / std. Mean=0, std=1 but unbounded. Outliers remain
- Min-max: (x - min) / (max - min). Bounded [0,1] but sensitive to outliers
- Rolling normalization: use only past window of data. Avoids look-ahead bias in live pipeline
- - Lag features: shift(n) on features so model sees only past data. Captures OFI persistence
- shift(1) on rolling: use only prior window for normalization. Avoids look-ahead bias
- model_df: clean DataFrame after dropna. Both features and target valid. Ready for training
- Queue imbalance vs trade imbalance: queue = sitting orders = intention. trade = executed orders = action
- Signal decay: OFI predicts returns best at short horizons. Decays over time. ACF measures this
- Liquidity sweep: institution sells to trigger retail stops, buys back cheaper. OFI spikes before price moves
- - Look-ahead audit: check row 0 of every lagged feature is NaN before dropna. After dropna, verify df_model.iloc[0] equals df_raw.iloc[0]
- dropna(): removes first row (NaN features from shift(1)) and last row (NaN target from shift(-1)). Applied after both feature lagging and target computation
- First valid prediction row: first timestamp where both features and target are valid. Always one row after raw row 0
- Lag validation: df_model.iloc[0][feature] must equal df_raw.iloc[0][feature]. Confirms shift worked correctly
