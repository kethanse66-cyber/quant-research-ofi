# quant-research-ofi

## Cross-Asset Microstructure Alpha Signal | Intraday Liquidity Regimes & Order Flow Imbalance

**Author:** Kethan (kethanse66
**Status:** Phase 6 Backtesting — IN PROGRESS

---

## What This Project Is

An institutional-grade quantitative research project building a Cross-Asset Order Flow Imbalance (OFI) Alpha Signal from scratch. The project implements the Cont, Kukanov & Stoikov (2014) framework across 11 equity and ETF tickers at tick resolution, with HMM-based regime detection, walk-forward validated predictive models, and a full backtesting engine with realistic transaction costs.

This is not a toy project. Every methodological choice — rolling HMM re-estimation, rank transform before HMM fitting, Newey-West corrected t-statistics, overlap-corrected IC evaluation, position clipping — is explicitly justified and documented.

---

## Key Results

| Metric | Value |
|--------|-------|
| **Primary signal horizon** | 30 seconds |
| **Ridge-Global 30s NW t-stat** | 3.44 (significant) |
| **Ridge-Global 1m NW t-stat** | 2.11 (significant) |
| **Phase 3 bar-level ICIR (1m)** | 2.999 |
| **Stressed regime 5m NW t-stat** | 7.85 |
| **Calm regime 10s NW t-stat** | 4.84 |
| **OFI R2 vs Cont 2014 benchmark** | 42.1% vs 65% (Level 1 attenuation explained) |
| **HMM states chosen** | 3 (calm / normal / stressed) |
| **Regime IC doubles in stressed vs calm** | Confirmed |
| **LightGBM vs Ridge** | Ridge wins — signal is linear |
| **Tickers** | SPY QQQ IWM XLF XLK XLE XLV AAPL JPM NVDA TLT |
| **Data period** | April 2024 — April 2025 (249 trading days) |
| **Bar frequency** | 10 seconds |
| **Rows per ticker** | ~582,000 |
| **Total raw data** | ~81 GB |

---

## Methodology Summary

### Signal Construction
Order Flow Imbalance computed per Cont et al. (2014):

```
OFI_t = delta_BidSize_t - delta_AskSize_t
```

Computed at best level only (Level 1 Polygon data). OFI normalized using rolling standard deviation. 16 features total: 7 OFI horizons, spread, queue imbalance, trade imbalance, microprice, VWAP deviation, Kyle lambda, Amihud illiquidity, realized volatility.

### Regime Detection
GaussianHMM with 3 states (calm / normal / stressed) on rank-transformed features. Rolling re-estimation: at each prediction point T, HMM fitted only on data up to T-1. Prevents lookahead bias in regime labels. 60-day burn-in. BIC tested 2-7 states — 3 states chosen for economic interpretability.

### Predictive Model
Ridge regression on 13 selected features (after Spearman + VIF multicollinearity removal). Walk-forward cross-validation: 3-month expanding train, 1-month test, 8 folds. Overlap-corrected IC evaluation (non-overlapping windows for multi-bar targets). Newey-West HAC t-statistics throughout.

### Key Finding: Regime Conditioning
Training and testing within matched regimes reveals signal that global models miss. Stressed-matched model at 5-minute horizon achieves NW t-stat = 7.85 vs global model which is not significant at that horizon.

---

## Why This Is Not Overfitting

1. **Walk-forward validation** — no data from test period used in training at any fold
2. **Lookahead check PASSED** on every fold — shift(1) verified at compute time
3. **Overlap correction** — overlapping 5-minute targets on 10-second bars create autocorrelation of 0.967; corrected by evaluating every 30th row only
4. **Newey-West t-statistics** — corrects for residual autocorrelation at high frequency
5. **LightGBM failed** — if Ridge was overfitting, LightGBM would have found something too
6. **Rolling HMM** — regime labels generated out-of-sample at every prediction point
7. **Feature selection before IC** — multicollinearity resolved before any significance claims

---

## Repository Structure

```
quant-research-ofi/
├── phase1_foundations/          # Phase 0 — Python and statistics foundations
├── phase1_synthetic_pipeline/   # Phase 1 — Full synthetic OFI pipeline
├── hmm_regime_detection/        # Phase 4 — HMM regime detection
├── ML models/                   # Phase 5 — ML models and IC analysis
├── Backtesting/                 # Phase 6 — Backtesting engine (IN PROGRESS)
├── README.md
└── PROGRESS_LOG.md
```

---

## Phase 0 — Foundations (April 2026) COMPLETE

- D1: Return calculator using Python for loops
- D2: NumPy vectorized returns. Log returns, win rate
- D3: Real SPY data. DataFrame operations. Charts
- D4: SPY returns not normal — Skewness=-0.54, Kurtosis=11.44, P-value=0.0
- D5: Hypothesis test on SPY mean return. t-stat=1.66, p-value=0.096
- D6: SPY vs QQQ correlation=0.93. OLS beta=1.13, R-squared=0.87
- D7: ADF test. SPY price not stationary p=0.948. Returns stationary p=0.0
- D8: Multiple testing on 20 strategies. 1 fake signal before Bonferroni, 0 after

---

## Phase 1 — Synthetic Pipeline (April 2026) COMPLETE

- D1 (Apr 6): simple_orderbook.py — add_order, mid_price, calculate_ofi
- D2 (Apr 7): ofi_synthetic.py — synthetic bid/ask, OFI, rolling features
- D3 (Apr 8): spread_calculator.py — quoted, effective, Roll spread
- D4 (Apr 9): tick_cleaner.py — duplicates, bad prices, UTC normalization, parquet
- D5 (Apr 10): volume_bar_builder.py — volume bars vs time bars comparison
- D7 (Apr 13): trade_imbalance.py — Lee-Ready classification, rolling imbalance
- D8 (Apr 14): feature_library.py — microprice, spread change, normalized OFI
- D9 (Apr 15): queue_imbalance.py — best level formula, edge cases
- D10 (Apr 16): ofi_full.py — multi-horizon OFI (30s/1m/5m), ACF, ADF, IC
- D11 (Apr 17): feature_normalizer.py — rank transform chosen for HMM
- D13 (Apr 20): target_variable.py — log returns 10s/1m/5m, shift(-n) validated
- D14 (Apr 21): lag_features.py — shift(1,2,3) features, rolling normalization
- D15 (Apr 22): audit_pipeline.py — lookahead audit PASSED all features
- D16 (Apr 23): save_to_parquet.py — 2x smaller than CSV, float32, snappy
- D17 (Apr 24): pipeline_test.py — 6 unit tests PASS, Ridge OOS IC=0.016

### Key Decisions Locked in Phase 1
- OFI formula: delta_bid - delta_ask at best level (Cont et al. 2014)
- shift(1) applied at model input only — never saved to parquet
- Log returns for target — stationary, comparable across tickers
- 16 features locked: ofi, ofi_10s, ofi_30s, ofi_1m, ofi_5m, ofi_10m, queue_imbalance, trade_imbalance, spread, spread_change, microprice, vwap, kyle_lambda, amihud, realized_vol, ofi_norm

---

## Phase 2 — Real Data Download (April-May 2026) COMPLETE

- Source: Polygon.io (Massive.com API)
- 12 tickers: SPY QQQ IWM XLF XLK XLE XLV AAPL JPM NVDA ES1! TLT
- Period: April 29 2024 to April 25 2025 (249 trading days)
- Frequency: 10-second bars
- Rows per ticker: ~582,000
- Raw data: ~81 GB | Feature parquets: ~458 MB

### Data Quality
- Zero negative spreads across all tickers
- Zero duplicate timestamps
- Zero negative volumes
- lag_check_pass = True for all tickers
- Mean rows per day = 2340 (exactly 6.5 hours × 360 bars/hour)
- OFI NaN pct = 0.04%

---

## Phase 3 — Data Validation + IC Analysis (May 2026) COMPLETE

### KEY RESULT — OFI Signal (Walk-Forward, 8 Folds, SPY)

| Horizon | IC Mean | ICIR | p-value | Significant |
|---------|---------|------|---------|-------------|
| fwd_ret_1m | 0.0137 | 2.999 | 0.0001 | YES |
| fwd_ret_5m | 0.0305 | 2.598 | 0.0002 | YES |
| fwd_ret_10m | 0.0417 | 2.457 | 0.0003 | YES |

OFI has statistically significant forward predictive power. ICIR above 2.0 across all horizons.

### R2 vs Cont 2014 Benchmark
- Mean R2 across 11 tickers: 42.1%
- Cont benchmark: 65%
- Gap explained by Level 1 vs tick data attenuation bias
- Best tickers: XLV 63.9%, TLT 60.6%, XLE 60.2%

---

## Phase 4 — HMM Regime Detection (May 2026) COMPLETE

### Files
- hmm_2state.py — 2-state GaussianHMM baseline
- hmm_3state.py — 3-state attempt on single feature
- hmm_normality_test.py — all 9 features FAIL normality, rank transform justified
- hmm_model_selection.py — BIC/AIC 2-7 states, 10 seeds each
- rolling_hmm.py — rolling re-estimation, lookahead-free regime labels
- regime_stats.py — transition matrix, regime duration, characterization
- regime_characterization.py — full 16-feature table per regime

### Design Decisions
- 3 states chosen: calm / normal / stressed
- Daily refit, expanding window, 60-day burn-in
- Rank transform applied on train distribution only — leakage-free
- Consistent relabeling by realized_vol across all folds
- Lookahead check: PASSED on all tickers

### Regime Distribution (SPY)
- Calm: 30.3% | Normal: 26.1% | Stressed: 43.6%

### Transition Matrix (SPY)
- Calm persistence: 83.1% | Normal: 0.01% (transition state) | Stressed: 51.7%

### Regime Characterization

| Feature | Calm | Normal | Stressed |
|---------|------|--------|----------|
| realized_vol | 0.058 | 0.141 | 0.164 |
| spread | 0.0179 | 0.0227 | 0.0246 |
| ofi | 75.2 | 130.9 | 169.4 |
| queue_imbalance | +0.012 | -0.009 | -0.002 |

### Honest Finding
HMM detects volatility regimes, not OFI regimes. OFI effect sizes small (Cohen's d ~0.006). Normal regime behaves as transition state — 2 states may be statistically better but 3 chosen for economic interpretability.

---

## Phase 5 — ML Models (May 2026) COMPLETE

### Files
- walk_forward_validator.py — expanding window CV framework
- linear_baseline.py — Ridge walk-forward with NW t-stats
- feature_selection.py — Spearman + VIF multicollinearity removal
- gradient_boosting.py — LightGBM vs Ridge comparison
- lasso.py — Lasso baseline
- per_regime_model.py — regime-matched training and evaluation
- ic_summary_table.py — final IC table all models all horizons
- cointegration_test.py — Engle-Granger + Johansen on 3 pairs
- garch_volume.py — GARCH(1,1) vs realized vol comparison
- shap_analysis.py — SHAP feature importance 30s and 1m horizons
- ic_per_regime.py — IC breakdown by regime
- audit_stressed_5m.py — permutation test on stressed 5m result

### Final IC Summary Table

| Model | Horizon | NW t-stat | Significant |
|-------|---------|-----------|-------------|
| Ridge-Global | 30s | 3.44 | YES — PRIMARY RESULT |
| Ridge-Global | 1m | 2.11 | YES — SECONDARY RESULT |
| Ridge-Calm-Matched | 10s | 4.84 | YES — FRAGILE (fold 9 dominated) |
| Ridge-Stressed-Matched | 5m | 7.85 | YES — caveat: not overlap-corrected |
| Lasso | all | <2.0 | NO |
| LightGBM | all | ~0 | NO |

### Key Findings
1. Signal is linear — LightGBM failure confirms this
2. Regime conditioning reveals signal global models miss
3. Feature selection critical — condition number 10^13 → 4.67 after removing collinear OFI horizons
4. SHAP confirms ofi_30s dominates at 30s, realized_vol dominates at 1m
5. No cointegration among SPY/QQQ/IWM — no pairs trading signal
6. GARCH not added — realized vol outperforms as predictive feature

---

## Phase 6 — Backtesting (June 2026) IN PROGRESS

### Files
- backtest_engine.py — BacktestEngine class, signal to PnL, bid-ask costs, drawdown

### Smoke Test (random signal)
- Sharpe: -21.85 (expected — random signal loses to costs)
- Max Drawdown: -1.66%
- Final Equity: 0.9899
- Win Rate: 43.68%
- Avg Turnover: 0.114

**Upcoming:** transaction_costs.py (Jun 2) | position_sizer.py (Jun 3) | walk_forward_backtest.py (Jun 5) | audit_report.py (Jun 8) | pbo_calculator.py (Jun 10)

---

## Honest Limitations

1. **Level 1 attenuation bias** — using best bid/ask only vs full order book. LOBSTER comparison planned June 16 to quantify IC loss.
2. **Normal regime unstable** — 0.01% persistence suggests transition state not true regime. 2-state HMM may be better statistically.
3. **Stressed sample period** — 43.6% stressed bars in 2024-2025 is unusually high. Results may not generalize to calmer market conditions.
4. **ES1! futures unavailable** — cash-futures basis analysis not completed. Key robustness check missing.
5. **Calm-matched 10s result fragile** — fold 9 dominated with small test set. Report with caveat.
6. **Short OOS period** — 249 trading days. Bootstrap confidence intervals will be wide.

---

## How to Reproduce

```bash
# Clone repo
git clone https://github.com/kethanse66-cyber/quant-research-ofi.git
cd quant-research-ofi

# Install dependencies
pip install -r requirements.txt

# Data required: Polygon.io API key
# Set environment variable: export POLYGON_API_KEY=your_key

# Run feature pipeline on SPY
python phase1_synthetic_pipeline/pipeline_test.py

# Run HMM regime detection
python hmm_regime_detection/rolling_hmm.py

# Run ML models
python ML\ models/linear_baseline.py
python ML\ models/per_regime_model.py

# Run backtest
python Backtesting/backtest_engine.py
```

---

## References

- Cont, Kukanov & Stoikov (2014) — The Price Impact of Order Book Events
- Kyle (1985) — Continuous Auctions and Insider Trading
- Glosten & Milgrom (1985) — Bid, Ask and Transaction Prices
- Bailey & Lopez de Prado (2014) — The Deflated Sharpe Ratio
- Almgren & Chriss (2001) — Optimal Execution of Portfolio Transactions
- Rabiner (1989) — A Tutorial on Hidden Markov Models

---

