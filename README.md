# Predictive Order Flow Imbalance: Cross-Asset Microstructure Alpha

**Author:** Kethan S E  
**Paper:** [SSRN — insert link]  
**Status:** Complete

---

## What this is

I built this project to answer one question: does Order Flow Imbalance actually predict future returns, or does it just explain what already happened?

The short answer is yes, it predicts — but the signal doesn't survive transaction costs at the frequency where it's strongest. That gap between statistical significance and economic viability is the main finding.

The project covers the full research pipeline: raw tick data download, feature engineering, HMM regime detection, walk-forward Ridge regression, and a backtesting engine with real transaction costs. Everything is built from scratch, not from existing frameworks.

---

## The numbers

| What | Result |
|---|---|
| Mean IC across 84 folds, 12 tickers | +0.0044 |
| Cross-asset t-statistic | 3.26 (p = 0.0016) |
| Ticker-clustered t-statistic | 4.02 (p = 0.0020) |
| OOS IC (March–April 2025) | +0.0022 |
| OOS net Sharpe | +0.246 |
| Gross Sharpe | +0.981 |
| Net Sharpe at 10-second execution | -1.726 |
| Cost-to-edge ratio at 10s | 164x — signal destroyed |
| Minimum viable holding period | 30 minutes (net Sharpe +0.114) |
| Only instrument with positive net Sharpe | NVDA (+0.14) |

---

## Why the signal doesn't work (and why that's the finding)

At 10-second bars, spread crossing costs run about 819 bps per day against a gross signal edge of roughly 5 bps per day. That's 164 times the edge. The signal is real — statistically, it holds up across 12 tickers, 84 folds, and an out-of-sample period — but you can't trade it at the frequency where it's most informative without destroying it through costs.

The three things that would fix this:
1. Hold positions for 30+ minutes instead of flipping every bar
2. Use limit orders instead of market orders (spread cost is 69% of total cost)
3. Only trade in stressed regimes where the signal IC is roughly 3x higher

I didn't deploy any of these in this version. They're the next research steps, not retrospective fixes.

---

## Data

- Source: Polygon.io (Massive.com)
- 12 instruments: SPY, QQQ, IWM, XLF, XLK, XLE, XLV, AAPL, JPM, NVDA, TLT, ES1!
- Period: April 2024 – April 2025 (249 trading days)
- Frequency: 10-second bars
- ~582,000 rows per ticker, ~7M bars total
- Zero data quality issues across all instruments

For the Level-1 vs Level-2 OFI comparison: Databento XNAS ITCH MBO data for NVDA, Feb–Mar 2025, 40 days, ~14M order book events per day.

---

## Methodology

**OFI formula** (Cont, Kukanov & Stoikov 2014):OFI_t = delta_BidSize_t - delta_AskSize_t at best price level
**Features:** 16 microstructure features, 13 retained after Spearman correlation + VIF filtering (condition number dropped from 10^13 to 4.67)

**Regime detection:** 2-state rolling HMM (calm/stressed) fitted daily on data up to T-1 only. Rank-transformed inputs. 60-day burn-in. Consistent state labelling by realized volatility.

**Signal research:** Walk-forward Ridge regression, 10 folds for SPY, 84 folds cross-asset. Non-overlapping IC evaluation. Newey-West HAC t-statistics throughout.

**Backtest:** Almgren-Chriss market impact model, real bid-ask spread from tick data, 0.10 bps commission. Transaction cost aware throughout.

**Robustness:** Deflated Sharpe Ratio, PBO-style CSCV, OOS test on held-out period, Fama-French 3-factor attribution (R² = 4% — strategy is largely idiosyncratic).

---

## What I checked so no one else has to

- Lookahead bias: shift(1) applied at model input time only, never saved to parquet. Audit confirms this on every fold.
- Overlapping targets: 5-minute forward returns on 10-second bars have lag-1 autocorrelation of 0.967. IC evaluated on non-overlapping windows only.
- Multicollinearity: longer OFI horizons are rolling sums of shorter ones. Condition number was 10^13 before feature selection. Fixed.
- HMM lookahead: regime at time T uses only data up to T-1. Rolling re-estimation, not full-sample.
- Multiple testing: Newey-West corrected t-statistics, PBO-style cross-validation, honest disclosure of non-significant month-level clustering result (p = 0.065).

---

## Repository structure


---

## References

- Cont, Kukanov & Stoikov (2014) — The Price Impact of Order Book Events
- Kyle (1985) — Continuous Auctions and Insider Trading
- Bailey & López de Prado (2014) — The Deflated Sharpe Ratio
- Almgren & Chriss (2001) — Optimal Execution of Portfolio Transactions
- Petersen (2009) — Estimating Standard Errors in Finance Panel Data Sets
