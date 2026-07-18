# Progress Log

## Phase 1 — Synthetic Pipeline (April 2026) ✓

Built the full OFI pipeline on synthetic data before touching real tick data.

Files: simple_orderbook.py | ofi_synthetic.py | spread_calculator.py | tick_cleaner.py | volume_bar_builder.py | trade_imbalance.py | feature_library.py | queue_imbalance.py | ofi_full.py | feature_normalizer.py | target_variable.py | lag_features.py | audit_pipeline.py | save_to_parquet.py | pipeline_test.py

Key decisions locked here:
- OFI = delta_bid - delta_ask at best level only (Cont et al. 2014)
- shift(1) applied at model input only, never saved to parquet
- Log returns for target variable
- 16 features: ofi family (6 horizons), spread, queue_imbalance, trade_imbalance, microprice, vwap, kyle_lambda, amihud, realized_vol, ofi_norm

---

## Phase 2 — Real Data (April–May 2026) ✓

Downloaded 12 tickers from Polygon.io. 249 trading days. ~582k rows per ticker. ~81GB raw data, 458MB in feature parquets.

Data quality: zero negative spreads, zero duplicate timestamps, zero negative volumes across all instruments.

---

## Phase 3 — IC Analysis (May 2026) ✓

OFI signal check on real data (SPY, 8 walk-forward folds):

| Horizon | IC Mean | ICIR | Significant |
|---|---|---|---|
| 1m | 0.0137 | 2.999 | YES |
| 5m | 0.0305 | 2.598 | YES |
| 10m | 0.0417 | 2.457 | YES |

Mean R² across 11 tickers: 42.1% (Cont 2014 benchmark: 65%). Gap attributed to Level-1 approximation.

---

## Phase 4 — HMM Regime Detection (May 2026) ✓

2-state rolling HMM (calm/stressed). Daily refit on data up to T-1 only. Rank-transformed inputs (all features fail normality tests). 60-day burn-in. Consistent relabelling by realized_vol.

SPY regime distribution: calm 27.1%, stressed 72.9%. Stressed regime persistence 91.4%.

Contemporaneous IC = 0.24, matching Cont et al. (2014). Forward IC near zero at 10-second resolution — signal predictive at longer lags only.

Honest finding: HMM detects volatility regimes, not OFI regimes. OFI effect size across regimes is small (Cohen's d ≈ 0.01).

---

## Phase 5 — ML Models (May 2026) ✓

Walk-forward Ridge regression across 10 folds (SPY) and 84 folds cross-asset. Non-overlapping IC evaluation. Newey-West HAC throughout.

Primary result: Ridge-Global 30s NW t-stat = 3.44 (7/10 folds positive, balanced test sets)
Secondary result: Ridge-Global 1m NW t-stat = 2.11
Cross-asset: mean IC = +0.0044, t = 3.26, p = 0.0016 (84 folds, 12 tickers)

LightGBM: near-zero IC everywhere — confirms signal is linear.
Lasso: not significant anywhere — signal is distributed across features, not sparse.

Feature selection: removed ofi, ofi_1m, ofi_5m (Spearman > 0.8). Condition number 10^13 → 4.67.

SHAP: ofi_30s dominates at 30s horizon, realized_vol at 1m. Not data mining — driven by what theory predicts.

---

## Phase 6 — Backtesting and Validation (June 2026) ✓

Cross-asset backtest: 84 folds, 12 tickers, April 2024 – April 2025.

Gross Sharpe: +0.981 | Net Sharpe: -1.726 | Survival rate: 35.7%

Signal does not survive transaction costs at 10-second execution. Spread crossing alone costs 819 bps/day against ~5 bps gross edge (164x). Minimum viable holding period: 30 minutes (net Sharpe +0.114).

NVDA is the only instrument with positive mean net Sharpe (+0.14). Post burn-in (Jan–Apr 2025), three instruments achieve positive net Sharpe: NVDA (+1.34), QQQ (+1.49), ES1! (+1.09).

OOS test (Mar–Apr 2025, never seen during development): OOS IC = +0.0022, OOS net Sharpe = +0.246.

Robustness: DSR computed, PBO-style CSCV (score 0.14, below random baseline 0.50), FF3 factor attribution (R² = 4%, alpha p = 0.767 — idiosyncratic returns).

Level-1 vs Level-2 OFI comparison using Databento MBO data (NVDA, 40 days): depth-aware measure shows 77% higher absolute IC at 5m horizon. Not statistically significant over 40-day window. Reported as exploratory.

Paper submitted to SSRN: https://dx.doi.org/10.2139/ssrn.7053198
