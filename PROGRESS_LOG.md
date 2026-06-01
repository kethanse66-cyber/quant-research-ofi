# PROGRESS LOG — quant-research-ofi


## Phase 0 — Foundations (April 2026) COMPLETE

### D1 — Return Calculator
**Learned:** Python for loops, lists, basic math operations
**Built:** return_calculator.py — daily returns, mean, variance, std using for loops
**Confused:** Nothing major

### D2 — NumPy Foundations
**Learned:** NumPy arrays, vectorized returns, log returns, indexing, conditional filtering, win rate, expected value, basic probability
**Built:** return_calculator_numpy.py — max, min, mean, std, win rate, log returns, no for loops
**Confused:** Log returns math, e and natural log. Mental math multiplication still slow

### D3 — Pandas Basics
**Learned:** DataFrame operations, pct_change, rolling mean, resample, isnull, loc, sort_values, dropna. Read COVID crash March 2020 in real data
**Built:** pandas_basics.py — SPY closing price chart, daily returns chart, rolling average chart, 5 worst and best days
**Confused:** Nothing major

### D4 — Descriptive Stats + SciPy
**Learned:** Skewness, kurtosis, normality test, histogram with normal curve overlay, density=True, np.linspace, stats.norm.pdf
**Built:** stats_report.py — proved SPY returns not normal three ways. Skewness=-0.54, Kurtosis=11.44, P-value=0.0
**Confused:** Nothing major

### D5 — Hypothesis Testing
**Learned:** Null hypothesis, t-statistic, p-value, ttest_1samp, f-strings, round()
**Built:** hypothesis_test.py — tested if SPY mean return differs from zero. t-stat=1.66, p-value=0.096, cannot confirm
**Confused:** Nothing major

### D6 — Correlation + OLS Regression
**Learned:** Correlation direction, SPY vs QQQ=0.93, OLS beta, R-squared, sm.add_constant, residual skew and kurtosis
**Built:** regression_analysis.py — SPY vs QQQ correlation, OLS regression, scatter plot. Beta=1.13, R-squared=0.87
**Confused:** Nothing major

### D7 — Time Series + Stationarity
**Learned:** Stationarity, ADF test, SPY price NOT stationary p=0.948, SPY returns stationary p=0.0, rolling mean and std
**Built:** stationarity_test.py — ADF test on SPY price and returns, rolling mean and std chart
**Confused:** Nothing major

### D8 — Multiple Testing Problem
**Learned:** Why many strategies produce fake signals, Bonferroni correction, new threshold = 0.05 / number of tests
**Built:** multiple_testing_demo.py — 20 random strategies, Bonferroni applied, 1 fake signal before, 0 after
**Confused:** Nothing major

---

## Phase 1 — Synthetic Pipeline (April 2026) COMPLETE

### D1 (Apr 6) — simple_orderbook.py
**Learned:** Limit order book structure, bid and ask dictionaries, bid-ask spread, mid price formula, OFI formula
**Built:** simple_orderbook.py — add_order, mid_price, calculate_ofi functions
**Confused:** Nothing major

### D2 (Apr 7) — ofi_synthetic.py
**Learned:** OFI using delta bid and delta ask, synthetic order book generation, rolling window features, OFI positive=buy pressure, OFI negative=sell pressure
**Built:** ofi_synthetic.py — synthetic bid/ask sizes, OFI calculation, spread, rolling OFI mean, sum, std
**Confused:** np.convolve vs pandas rolling. Rolling window meaning in ticks vs days

### D3 (Apr 8) — spread_calculator.py
**Learned:** Quoted spread, effective spread, Roll spread, negative autocorrelation reveals bid-ask bounce, mid price
**Built:** spread_calculator.py — quoted spread, effective spread, Roll spread, comparison chart
**Confused:** Nothing major

### D4 (Apr 9) — tick_cleaner.py
**Learned:** Tick data cleaning pipeline, duplicate timestamps, missing price handling, bad price filtering, zero volume filtering, timestamp normalization, UTC conversion, resampling to fill missing timestamps
**Built:** tick_cleaner.py — synthetic tick data, duplicate injection, missing value handling, bad price removal, drop zero volume, resample to 1-second grid, UTC normalization, parquet save
**Confused:** Difference between tz_localize vs tz_convert, when to resample vs forward fill

### D5 (Apr 10) — volume_bar_builder.py
**Learned:** Time bars vs volume bars difference, OHLC construction from ticks, volume bars normalize market activity, quote data vs trade data difference. Polygon gives two separate files — quotes for OFI, trades for price and volume
**Built:** volume_bar_builder.py — volume bars with VWAP and total volume, time bars with OHLC for comparison
**Confused:** Nothing major

### D7 (Apr 13) — trade_imbalance.py
**Learned:** Lee-Ready rule to classify trades as buy or sell, volume-weighted trade imbalance, rolling window imbalance, why we use volume not just count, direction +1/-1/0 meaning
**Built:** trade_imbalance.py — classify_trade function, trade_imbalance function, rolling buy/sell volume
**Confused:** Nested np.where syntax, why direction=0 at mid price. bid_size and ask_size are orders waiting in book not volume traded

### D8 (Apr 14) — feature_library.py
**Learned:** Microprice weighted by bid/ask size, spread change as liquidity signal, OFI normalization using rolling standard deviation, why normalization stabilizes scale across time
**Built:** feature_library.py — compute_ofi, microprice, spread change, normalized OFI
**Confused:** Why normalize OFI and not use raw OFI

### D9 (Apr 15) — queue_imbalance.py
**Learned:** Queue imbalance at best level, why Level 1 only matters for Polygon data, edge case handling with np.where
**Built:** queue_imbalance.py — queue_imbalance_best function, edge case tests
**Confused:** Nothing major

### D10 (Apr 16) — ofi_full.py
**Learned:** Multi-horizon OFI at 30s, 1min, 5min bars, autocorrelation, ADF stationarity test, IC, lagged OFI avoids lookahead bias
**Built:** ofi_full.py — tick-level OFI, horizon resampling, ACF analysis, ADF test, IC calculation, visualization dashboard
**Confused:** Why shorter or longer time horizons give different IC results

### D11 (Apr 17) — feature_normalizer.py
**Learned:** Three normalization methods — rank transform, z-score, min-max. Rolling window versions avoid lookahead bias. Rank transform preferred before HMM because OFI is fat-tailed — rank removes distributional shape entirely
**Built:** feature_normalizer.py — zscore_normalize, minmax_normalize, rank_transform, normalization_audit
**Confused:** Nothing major

### D13 (Apr 20) — target_variable.py
**Learned:** Log returns at 3 horizons. shift(-n) works in rows not seconds. 1 row = 10 seconds so 1min = shift(-6), 5min = shift(-30). Last N rows get NaN. Index frequency check confirms all rows exactly 10s apart.
**Built:** target_variable.py — compute_log_returns function, parameterized horizons, index frequency validation
**Confused:** NaN concept took time — understood

### D14 (Apr 21) — lag_features.py
**Learned:** Lag features shift every feature back by 1,2,3 rows so model only sees past data at time T. shift(1) on rolling std and mean avoids lookahead bias in normalization. dropna on lag_cols + target together.
**Built:** lag_features.py — create_lag_features, shift(1) rolling normalization, explicit dropna
**Results:** model_df has 470 rows after dropping NaN warmup rows. First valid prediction row = 9:34:00
**Confused:** Nothing major

### D15 (Apr 22) — audit_pipeline.py
**Learned:** Lookahead audit logic — shift(1) on features means row 0 of df_model must equal raw row 0 of df_raw. After shift(1) and dropna(), first valid prediction row is 09:31am not 09:30am.
**Built:** audit_pipeline.py — feature lag audit, row0 NaN check, lag validation, dropna fix
**Results:** All features PASS lookahead audit
**Confused:** shift(1) direction took time — understood

### D16 (Apr 23) — save_to_parquet.py
**Learned:** Parquet saves by column not row — reads faster, uses less disk than CSV. float32 halves memory vs float64. snappy compression built into Parquet write.
**Built:** save_to_parquet.py — parquet save/load with UTC timestamps, float32 optimization, benchmark
**Results:** Parquet 2x smaller (9.5MB vs 19.8MB), writes 2x faster
**Confused:** Nothing major

### D17 (Apr 24) — pipeline_test.py
**Learned:** Full OFI formula from Cont et al. 2014 — handles bid/ask price changes and queue depletion. Unit tests with assert catch silent bugs. IC measures per-feature predictive power.
**Built:** pipeline_test.py — full end-to-end pipeline test, 6 unit tests, IC analysis, Ridge baseline
**Results:** 6 unit tests PASS. Ridge OOS IC=0.016, R2=-0.17. Pipeline completes in 22 seconds
**Key Finding:** Negative R2 on synthetic data expected — no real institutional order flow in random data

---

## Phase 2 — Real Data Download (April-May 2026) COMPLETE

### D19-D21 — data_pipeline.py + downloads
**Learned:** Massive.com Futures API pagination via next_url cursor. Async download for multiple tickers simultaneously.
**Built:** data_pipeline.py — Massive.com API download, pagination handling, parquet save
**Results:** 12 tickers downloaded. 249 trading days. ~81GB raw data. Zero data quality issues.
**Key Finding:** Mean rows per day = 2340 exactly (6.5 hours × 360 bars/hour) — confirms 10-second bar frequency

---

## Phase 3 — Data Validation + IC Analysis (May 2026) COMPLETE

### validate_data.py + ic_analysis.py + cross_asset_ofi.py
**Learned:** Real data validation pipeline. Walk-forward IC on actual tick data. Cross-asset OFI signal construction.
**Built:** validate_data.py | plot_ofi.py | ic_analysis.py | cross_asset_ofi.py | feature_test.py
**Results:**
- Zero negative spreads, zero duplicate timestamps, zero negative volumes across all tickers
- OFI IC at 1m: mean=0.0137, ICIR=2.999, p=0.0001 — SIGNIFICANT
- OFI IC at 5m: mean=0.0305, ICIR=2.598, p=0.0002 — SIGNIFICANT
- OFI IC at 10m: mean=0.0417, ICIR=2.457, p=0.0003 — SIGNIFICANT
- Mean R2 vs Cont 2014: 42.1% (benchmark 65%) — gap from Level 1 attenuation
**Key Finding:** Signal exists in real data. ICIR above 2.0 at all horizons tested.

---

## Phase 4 — HMM Regime Detection (May 2026) COMPLETE

### D1 (May 4) — hmm_2state.py
**Learned:** HMM has 3 components — states (hidden regimes), emissions (what each state looks like), transitions (how states switch). Baum-Welch learns parameters. Viterbi labels each time point.
**Built:** hmm_2state.py — GaussianHMM 2 states on SPY realized volatility
**Results:** Low Vol mean=0.0718, High Vol mean=0.1998. Transition matrix 99.72% persistence.
**Confused:** Nothing major
**Key Finding:** Volatility regimes are sticky — once stressed, tends to stay stressed

### D2 (May 5) — hmm_3state.py
**Learned:** Adding a third state does not always help. When two states have nearly identical means, HMM is splitting noise not finding real regimes. Need more features.
**Built:** hmm_3state.py — GaussianHMM 3 states on SPY realized volatility
**Results:** Low Vol mean=0.0718 (34.9%), Medium Vol mean=0.0719 (34.9%), High Vol mean=0.1998 (30.2%)
**Key Finding:** Low and Medium Vol nearly identical — 3 states not justified on realized vol alone

### D3 (May 6) — hmm_normality_test.py
**Learned:** GaussianHMM assumes normal emissions. Financial data almost never satisfies this. Rank transform converts values to percentile ranks 0-1 — removes distributional shape entirely.
**Built:** hmm_normality_test.py — normaltest on 9 features before and after rank transform
**Results:** All 9 features FAIL normality. realized_vol skewness=5.96 kurtosis=65.6. amihud skewness=70.5 kurtosis=4972. After rank transform skewness≈0 for all features.
**Key Finding:** Never assume normality in financial data. Always test and document.

### D4 (May 7) — hmm_model_selection.py
**Learned:** BIC and AIC penalize model complexity. BIC heavier penalty than AIC. Running multiple random seeds per state count prevents convergence to local minima.
**Built:** hmm_model_selection.py — BIC/AIC for 2-7 states with 10 seeds each
**Results:** BIC decreasing through 7 states. Largest marginal improvement at 2→3 transition.
**Key Finding:** Mathematical optimum BIC=7 vs research choice 3 states is deliberate tradeoff. Interpretability and stability justify choosing 3.

### D5 (May 11) — rolling_hmm.py
**Learned:** Full-sample HMM uses future data to label past regimes — introduces lookahead bias in signal research. Rolling re-estimation: fit HMM on data up to T-1, predict regime at T.
**Built:** rolling_hmm.py — expanding window rolling HMM, 60-day burn-in, consistent relabeling by realized_vol
**Results:** Lookahead check PASSED on all 11 tickers. First valid prediction timestamp confirmed.
**Confused:** Consistent relabeling required — HMM state numbering is arbitrary, must sort by realized_vol mean each fit
**Key Finding:** Rolling HMM is non-negotiable. Citadel and Two Sigma will ask this directly.

### D6 (May 12) — regime_ofi_ic.py
**Learned:** Splitting data by regime before computing IC reveals signal differences invisible in global analysis. Contemporaneous IC measures immediate OFI impact, not predictive power.
**Built:** regime_ofi_ic.py — IC by regime for OFI vs forward returns
**Results:** Now IC (contemporaneous) = 0.24 across all regimes — matches Cont et al. 2014 exactly. Forward IC near zero at 10s resolution — OFI not predictive at bar level.
**Key Finding:** OFI is contemporaneous not predictive at 10-second resolution. Predictive power emerges at 30s+ after multi-horizon aggregation and feature combination.

### D7 (May 13) — regime_stats.py
**Learned:** Transition matrix quantifies regime persistence. Average duration shows how long market stays in each state.
**Built:** regime_stats.py — transition matrix, average duration per regime, regime characterization table
**Results:** Calm persistence 83.1%. Normal persistence 0.01% — behaves as transition state. Stressed persistence 51.7%. Calm avg duration 5.9 bars. Stressed avg duration 2.1 bars.
**Key Finding:** Normal regime is not a true regime — it is a transition state between calm and stressed.

### D8 (May 14) — regime_characterization.py
**Learned:** Cohen's d measures effect size — how different two distributions actually are regardless of sample size.
**Built:** regime_characterization.py — full 16-feature characterization per regime, Mann-Whitney + Cohen's d
**Results:** 14/16 features significantly different calm vs stressed. Large effect: realized_vol (d=1.21). Small effect: all OFI features (d~0.006).
**Key Finding:** HMM detects volatility regimes, not OFI regimes. OFI effect sizes small but signal still regime-dependent.

---

## Phase 5 — ML Models (May 2026) COMPLETE

### D1 (May 18) — walk_forward_validator.py
**Learned:** Overlapping targets produce fake IC. fwd_ret_5m on 10s bars has autocorrelation 0.967 — adjacent rows share 29 of 30 bars. Overlap correction: evaluate every 30th row only for 5m targets.
**Built:** walk_forward_validator.py — expanding window, 8 folds, overlap correction, lookahead assertion every fold
**Results:** 8 folds clean. Naive persistence IC = 0.95 before correction, drops to near zero after.
**Confused:** Overlap correction direction — understood after thinking through what fwd_ret_5m actually measures
**Key Finding:** Overlapping targets are a silent bug that inflates IC dramatically. Always correct.

### D2 (May 19) — linear_baseline.py
**Learned:** Newey-West HAC corrects for autocorrelated residuals. At 10s bar frequency with multi-bar targets, standard errors without HAC are unreliable. Regular t-stat=0.303, NW t-stat=0.543 — direction same, magnitude different.
**Built:** linear_baseline.py — Ridge walk-forward, overlap-corrected IC, NW t-stats, fold-by-fold results
**Results:** Ridge-Global 30s NW t-stat=3.44 SIGNIFICANT. Ridge-Global 1m NW t-stat=2.11 SIGNIFICANT. 5m and 10m not significant.
**Key Finding:** OFI signal exists at 30s and 1m horizons. Decays beyond 5 minutes. Consistent with microstructure theory.

### D3 (May 20) — feature_selection.py
**Learned:** OFI horizons nearly perfectly correlated at 10s bars — longer horizon OFI is rolling sum of shorter. Condition number 10^13 means Ridge cannot distinguish individual contributions. VIF above 10 = redundant feature.
**Built:** feature_selection.py — Spearman correlation filter (threshold 0.8) + VIF filter (threshold 10)
**Results:** Removed ofi, ofi_1m, ofi_5m. 16 → 13 features. Condition number 10^13 → 4.67. All VIF below 4.
**Key Finding:** Multicollinearity was severe. Feature selection non-negotiable before any IC estimation.

### D4 (May 21) — gradient_boosting.py
**Learned:** LightGBM failing where Ridge succeeds confirms linear relationship. Nonlinear model needs nonlinear signal. OFI price impact is theoretically linear per Cont et al.
**Built:** gradient_boosting.py — LightGBM walk-forward, SHAP values, IC comparison vs Ridge
**Results:** LightGBM NW t-stat near zero all horizons. Ridge wins at 30s and 1m.
**Key Finding:** Signal is linear. Confirms Cont et al. 2014. Not data mining — theoretically motivated.

### D5 (May 22) — per_regime_model.py
**Learned:** Training on global data mixes regimes and dilutes signal. Matched evaluation: train on regime X, test on regime X only. Mixed evaluation: train on regime X, test on all regimes.
**Built:** per_regime_model.py — separate Ridge per regime, matched and mixed evaluation, permutation test
**Results:** Stressed-matched 30s NW t-stat=3.48. Stressed-matched 5m NW t-stat=7.85. Calm-matched 10s NW t-stat=4.84.
**Confused:** Why stressed 5m is stronger than stressed 30s — understood: in volatile markets institutional flow is more persistent
**Key Finding:** Regime conditioning reveals signal global models miss entirely. HMM regime labels are economically meaningful.

### D6 (May 23) — ic_summary_table.py
**Learned:** Fold stability matters — one dominant fold with small test set makes result fragile. Overlap correction reduces matched regime samples — tradeoff between correction and sample size.
**Built:** ic_summary_table.py — final IC table mixed and matched, all 5 models, all 5 horizons
**Results:** Two robust primary results: Ridge-Global 30s NW t-stat=3.44, Ridge-Global 1m=2.11. Calm-matched 10s fragile — fold 9 IC=0.0746 on only 1893 rows. Stressed 5m secondary finding with overlap caveat.
**Key Finding:** Most robust finding is Ridge-Global 30s. 7/10 folds positive, large balanced test sets. This is the paper headline result.

### D7 (May 25) — cointegration_test.py
**Learned:** Cointegration requires stationary linear combination of non-stationary series. Non-stationary spread not suitable as feature even with marginal IC. ES1! unavailable — planned robustness check not completed.
**Built:** cointegration_test.py — Engle-Granger + Johansen on SPY/QQQ, SPY/IWM, QQQ/IWM. Rolling hedge ratio with 60-day window, shift(1) for lookahead prevention.
**Results:** All Engle-Granger p-values above 0.05. No cointegration detected. SPY-QQQ spread marginal OOS IC=0.015 at 30s (p=0.004) but non-stationary — not added to pipeline.
**Key Finding:** No pairs trading signal in this universe. Feature set unchanged at 13.

### D8 (May 26) — garch_vol.py
**Learned:** Alpha+beta above 1 = IGARCH — volatility shocks never decay. 2024-2025 sample unusually stressed (43.6% stressed bars) explains this behaviour.
**Built:** garch_vol.py — GARCH(1,1) on SPY 1-minute returns, conditional vol vs realized vol IC comparison
**Results:** omega=0.029, alpha=0.806, beta=0.196, alpha+beta=1.003. Realized vol IC at 30s=0.0023 vs GARCH IC=0.0019. GARCH not added.
**Key Finding:** Realized volatility outperforms GARCH as predictive feature. Near-IGARCH documents unusual market conditions in sample period.

### D9 (May 27) — shap_analysis.py
**Learned:** SHAP values tell you which features actually drive model predictions — not just which are correlated with outcome. Feature importance is horizon-dependent: different features matter at different timescales.
**Built:** shap_analysis.py — SHAP waterfall and bar plots at 30s and 1m horizons
**Results:** 30s horizon: ofi_30s rank 1, ofi_norm rank 2, realized_vol rank 3. 1m horizon: realized_vol rank 1, microprice_ret rank 2. kyle_lambda, amihud, queue_imbalance negligible both horizons.
**Key Finding:** Signal driven by what theory predicts — order flow at short horizons, volatility context at longer horizons. Not data mining.

---

## Phase 6 — Backtesting (June 2026) IN PROGRESS

### D1 (Jun 1) — backtest_engine.py
**Learned:** Drawdown must be computed on equity curve cumprod not cumsum — cumsum understates drawdowns in compounding strategies. Turnover is position change per bar not PnL change. Position must be clipped to prevent leverage blowup. Run backtest once at init — never recompute inside every method.
**Built:** backtest_engine.py — BacktestEngine class. Signal in, PnL out. Bid-ask spread crossing costs. Commission. Position clipped to max_position. Equity curve drawdown. Avg turnover correct formula.
**Results:** Smoke test random signal: Sharpe=-21.85, MaxDD=-1.66%, FinalEquity=0.9899, WinRate=43.68%, AvgTurnover=0.114
**Confused:** Nothing major
**Key Finding:** Random signal loses to transaction costs as expected. Confirms engine correctly penalizes high turnover noise strategies. Real OFI signal on SPY should show positive Sharpe after costs.

---

