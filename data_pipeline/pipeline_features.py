import pandas as pd
import numpy as np
import glob
import os
import sys
from datetime import datetime, time

# =============================================================================
# pipeline_features.py — FINAL VERSION
#
# All fixes applied:
# 1. Session filter uses dt.time — fast
# 2. Condition filter exact parsing — no false matches
# 3. INCLUDE_EMPTY per ticker — update from audit_conditions.py output
# 4. Lee-Ready tick-level with per-ticker tolerance
# 5. 10-second bars
# 6. ofi_norm uses rolling std — correct denominator
# 7. Forward returns use log returns
# 8. Realized vol annualisation corrected — 3600 not 360
# 9. Kyle Lambda with min_periods — stable estimates
# 10. Error logging saves to file
#
# USAGE: python3 pipeline_features.py SPY
# OUTPUT: ~/quant-research-ofi/features/SPY_features.parquet
# =============================================================================

RAW_DIR      = "/home/kethanse66/quant-research-ofi/raw_parquet"
FEATURES_DIR = "/home/kethanse66/quant-research-ofi/features"
REPORTS_DIR  = "/home/kethanse66/quant-research-ofi/reports"
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR,  exist_ok=True)

SESSION_START = time(9, 30)
SESSION_END   = time(16, 0)

BAR_SIZE     = "10s"
KYLE_MIN_OBS = 180   # 30 min of 10s bars
VOL_WINDOW   = 180   # 30 min of 10s bars

# Per-ticker merge_asof tolerance
# Liquid: 500ms | ETFs: 1s | Less liquid: 2s
QUOTE_TOLERANCE = {
    "SPY":  pd.Timedelta("500ms"),
    "QQQ":  pd.Timedelta("500ms"),
    "AAPL": pd.Timedelta("500ms"),
    "NVDA": pd.Timedelta("500ms"),
    "JPM":  pd.Timedelta("500ms"),
    "IWM":  pd.Timedelta("1s"),
    "XLF":  pd.Timedelta("1s"),
    "XLK":  pd.Timedelta("1s"),
    "XLE":  pd.Timedelta("1s"),
    "XLV":  pd.Timedelta("1s"),
    "TLT":  pd.Timedelta("2s"),
}
DEFAULT_TOLERANCE = pd.Timedelta("1s")

# FIX 3 — Per-ticker INCLUDE_EMPTY
# Update these values after running audit_conditions.py for each ticker
# True = include trades with empty/missing condition codes
# False = exclude them
INCLUDE_EMPTY_BY_TICKER = {
    "SPY":  False,   # update after audit
    "QQQ":  False,   # update after audit
    "IWM":  False,   # update after audit
    "XLF":  False,   # update after audit
    "XLK":  False,   # update after audit
    "XLE":  False,   # update after audit
    "XLV":  False,   # update after audit
    "AAPL": False,   # update after audit
    "JPM":  False,   # update after audit
    "NVDA": False,   # update after audit
    "TLT":  False,   # update after audit
}

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def log_error(error_log_path, date, error):
    """FIX 10 — Save errors to file for later review"""
    with open(error_log_path, "a") as f:
        f.write(f"{datetime.now().strftime('%H:%M:%S')} | {date} | {type(error).__name__} | {error}\n")

# =============================================================================
# LOAD QUOTES
# =============================================================================

def load_quotes(ticker, date):
    path = os.path.join(RAW_DIR, f"{ticker}_quotes_{date}.parquet")
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["sip_timestamp"], unit="ns", utc=True)
    df["ts"] = df["ts"].dt.tz_convert("America/New_York")

    t  = df["ts"].dt.time
    df = df[(t >= SESSION_START) & (t <= SESSION_END)]
    df = df[(df["bid_price"] > 0) & (df["ask_price"] > 0)]
    df = df[df["ask_price"] >= df["bid_price"]]
    df = df.sort_values("ts").reset_index(drop=True)
    return df

# =============================================================================
# LOAD TRADES
# =============================================================================

def parse_conditions(cond_str):
    """Exact parsing — no substring match. Returns list of ints."""
    try:
        cleaned = str(cond_str).replace("[","").replace("]","").strip()
        if not cleaned or cleaned in ("nan","None"):
            return []
        return [int(x.strip()) for x in cleaned.split(",")
                if x.strip().lstrip("-").isdigit()]
    except:
        return []

def is_valid_trade(cond_str, include_empty):
    """
    Valid trade if:
    - conditions empty/missing AND include_empty is True, OR
    - condition code 12 is present
    """
    cond_clean = str(cond_str).strip()
    if cond_clean in ("", "nan", "None", "[]"):
        return include_empty
    codes = parse_conditions(cond_clean)
    if not codes:
        return include_empty
    return 12 in codes or 37 in codes

def load_trades(ticker, date):
    path = os.path.join(RAW_DIR, f"{ticker}_trades_{date}.parquet")
    if not os.path.exists(path):
        return None

    include_empty = INCLUDE_EMPTY_BY_TICKER.get(ticker, True)

    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["sip_timestamp"], unit="ns", utc=True)
    df["ts"] = df["ts"].dt.tz_convert("America/New_York")

    t  = df["ts"].dt.time
    df = df[(t >= SESSION_START) & (t <= SESSION_END)]

    df["valid"] = df["conditions"].apply(lambda x: is_valid_trade(x, include_empty))
    df = df[df["valid"]].drop(columns=["valid"])
    df = df[(df["price"] > 0) & (df["size"] > 0)]
    df = df.sort_values("ts").reset_index(drop=True)
    return df

# =============================================================================
# LEE-READY TICK-LEVEL
# =============================================================================

def lee_ready_tick_level(quotes, trades, ticker):
    if trades is None or len(trades) == 0:
        return None

    tolerance = QUOTE_TOLERANCE.get(ticker, DEFAULT_TOLERANCE)

    q = quotes[["ts","bid_price","ask_price"]].copy()
    q["mid"] = (q["bid_price"] + q["ask_price"]) / 2
    q = q.sort_values("ts")

    t = trades[["ts","price","size"]].copy()
    t = t.sort_values("ts")

    merged = pd.merge_asof(
        t, q[["ts","mid","bid_price","ask_price"]],
        on="ts", direction="backward", tolerance=tolerance
    )
    merged = merged.dropna(subset=["mid"])

    # Lee-Ready with quote tiebreaker
    merged["trade_sign"] = np.where(
        merged["price"] > merged["mid"],  1,
        np.where(merged["price"] < merged["mid"], -1,
        np.where(merged["price"] >= merged["ask_price"],  1,
        np.where(merged["price"] <= merged["bid_price"], -1, 0)))
    )
    merged["signed_size"] = merged["trade_sign"] * merged["size"]
    return merged

# =============================================================================
# BUILD OFI — Cont Kukanov Stoikov 2014
# =============================================================================

def build_ofi_raw(quotes):
    df   = quotes[["ts","bid_price","ask_price","bid_size","ask_size"]].copy()
    prev = df.shift(1)

    df["delta_bid"] = np.where(
        df["bid_price"] > prev["bid_price"],   df["bid_size"],
        np.where(
        df["bid_price"] == prev["bid_price"],  df["bid_size"] - prev["bid_size"],
        -prev["bid_size"])
    )
    df["delta_ask"] = np.where(
        df["ask_price"] < prev["ask_price"],   df["ask_size"],
        np.where(
        df["ask_price"] == prev["ask_price"],  df["ask_size"] - prev["ask_size"],
        -prev["ask_size"])
    )
    df["ofi_raw"] = df["delta_bid"] - df["delta_ask"]
    return df

# =============================================================================
# BUILD ALL FEATURES ON 10s BARS
# =============================================================================

def build_features(ofi_df, signed_trades):
    df = ofi_df.set_index("ts")

    bars = df.resample(BAR_SIZE).agg({
        "ofi_raw":   "sum",
        "bid_price": "last",
        "ask_price": "last",
        "bid_size":  "last",
        "ask_size":  "last",
    })

    bars[["bid_price","ask_price","bid_size","ask_size"]] = \
        bars[["bid_price","ask_price","bid_size","ask_size"]].ffill()
    bars = bars.dropna(subset=["bid_price","ask_price"])

    # OFI at multiple horizons (1 bar = 10s)
    bars["ofi"]     = bars["ofi_raw"]
    bars["ofi_10s"] = bars["ofi_raw"].rolling(1).sum()
    bars["ofi_30s"] = bars["ofi_raw"].rolling(3).sum()
    bars["ofi_1m"]  = bars["ofi_raw"].rolling(6).sum()
    bars["ofi_5m"]  = bars["ofi_raw"].rolling(30).sum()
    bars["ofi_10m"] = bars["ofi_raw"].rolling(60).sum()

    # FIX 6 — ofi_norm uses rolling std not size denominator
    # Normalises OFI_1m by its own recent volatility — scale-free
    ofi_std          = bars["ofi_1m"].rolling(120, min_periods=20).std()
    bars["ofi_norm"] = bars["ofi_1m"] / ofi_std.replace(0, np.nan)

    # Spread
    bars["spread"]        = bars["ask_price"] - bars["bid_price"]
    bars["spread_change"] = bars["spread"].diff()

    # Queue imbalance
    total_size = bars["bid_size"] + bars["ask_size"]
    bars["queue_imbalance"] = (bars["bid_size"] - bars["ask_size"]) / \
                               total_size.replace(0, np.nan)

    # Microprice — weighted mid using queue sizes
    bars["microprice"] = (bars["bid_price"] * bars["ask_size"] +
                          bars["ask_price"] * bars["bid_size"]) / \
                          total_size.replace(0, np.nan)

    # Mid price
    bars["mid"] = (bars["bid_price"] + bars["ask_price"]) / 2

    # FIX 8 — correct annualisation: 252 days * 6.5 hours * 3600 seconds
    # At 10s bars: 360 bars per hour, 2340 bars per day
    bars["log_ret"]      = np.log(bars["mid"] / bars["mid"].shift(1))
    bars["realized_vol"] = bars["log_ret"].rolling(VOL_WINDOW, min_periods=30).std() * \
                           np.sqrt(252 * 6.5 * 360)

    # Trade features
    if signed_trades is not None and len(signed_trades) > 0:
        t = signed_trades.set_index("ts")
        trade_bars = t.resample(BAR_SIZE).agg({
            "signed_size": "sum",
            "size":        "sum",
            "price":       "last",
        })

        bars = bars.join(trade_bars, how="left")
        bars["signed_size"] = bars["signed_size"].fillna(0)
        bars["size"]        = bars["size"].fillna(0)
        bars["price"]       = bars["price"].ffill()

        # Trade imbalance rolling 1 min (6 bars)
        buy_vol  = bars["signed_size"].clip(lower=0).rolling(6).sum()
        sell_vol = bars["signed_size"].clip(upper=0).abs().rolling(6).sum()
        total_tv = buy_vol + sell_vol
        bars["trade_imbalance"] = (buy_vol - sell_vol) / total_tv.replace(0, np.nan)

        # VWAP rolling 5 min (30 bars)
        bars["vwap"] = (bars["price"].ffill() * bars["size"]).rolling(30).sum() / \
                        bars["size"].rolling(30).sum().replace(0, np.nan)

        # Kyle Lambda — price impact per unit signed flow
        # NOTE: interpret carefully — noisy at 10s level
        # Use as regime proxy not raw signal
        delta_mid  = bars["mid"].diff()
        signed_vol = bars["signed_size"]
        cov = delta_mid.rolling(KYLE_MIN_OBS, min_periods=KYLE_MIN_OBS).cov(signed_vol)
        var = signed_vol.rolling(KYLE_MIN_OBS, min_periods=KYLE_MIN_OBS).var()
        bars["kyle_lambda"] = cov / var.replace(0, np.nan)

        # Amihud illiquidity — |return| per dollar volume
        # NOTE: compare within ticker only — not comparable across tickers directly
        bars["amihud"] = bars["log_ret"].abs().rolling(30).sum() / \
                         bars["size"].rolling(30).sum().replace(0, np.nan)
    else:
        bars["trade_imbalance"] = np.nan
        bars["vwap"]            = np.nan
        bars["kyle_lambda"]     = np.nan
        bars["amihud"]          = np.nan

    return bars

# =============================================================================
# FORWARD RETURNS + LAG
# =============================================================================

def add_targets_and_lag(bars):
    # FIX 7 — log returns for forward targets
    # Log returns are additive and symmetric — correct for regression targets
    bars["fwd_ret_1m"]  = np.log(bars["mid"].shift(-6)  / bars["mid"])
    bars["fwd_ret_5m"]  = np.log(bars["mid"].shift(-30) / bars["mid"])
    bars["fwd_ret_10m"] = np.log(bars["mid"].shift(-60) / bars["mid"])

    # Lag all features by 1 bar — prevent look-ahead bias
    # Signal at time T uses only information available before T
    feature_cols = [
        "ofi","ofi_10s","ofi_30s","ofi_1m","ofi_5m","ofi_10m",
        "ofi_norm","queue_imbalance","spread","spread_change",
        "microprice","realized_vol","trade_imbalance","vwap",
        "kyle_lambda","amihud"
    ]
    for col in feature_cols:
        if col in bars.columns:
            bars[col] = bars[col].shift(1)

    return bars

# =============================================================================
# PROCESS ONE DAY
# =============================================================================

def process_day(ticker, date):
    quotes = load_quotes(ticker, date)
    if quotes is None or len(quotes) < 1000:
        return None

    trades        = load_trades(ticker, date)
    signed_trades = lee_ready_tick_level(quotes, trades, ticker)
    ofi_df        = build_ofi_raw(quotes)
    bars          = build_features(ofi_df, signed_trades)
    bars          = add_targets_and_lag(bars)

    bars["ticker"] = ticker
    bars["date"]   = date

    keep_cols = [
        "ticker","date",
        "ofi","ofi_10s","ofi_30s","ofi_1m","ofi_5m","ofi_10m",
        "ofi_norm","queue_imbalance","spread","spread_change",
        "microprice","realized_vol","trade_imbalance","vwap",
        "kyle_lambda","amihud",
        "fwd_ret_1m","fwd_ret_5m","fwd_ret_10m"
    ]

    available    = [c for c in keep_cols if c in bars.columns]
    bars         = bars[available].reset_index()
    feature_only = [c for c in available if c not in ["ticker","date","ts"]]
    bars         = bars.dropna(subset=feature_only, how="all")
    return bars

# =============================================================================
# MAIN
# =============================================================================

def main(ticker):
    log(f"Starting feature pipeline for {ticker}")
    log(f"Tolerance:     {QUOTE_TOLERANCE.get(ticker, DEFAULT_TOLERANCE)}")
    log(f"Bar size:      {BAR_SIZE}")
    log(f"INCLUDE_EMPTY: {INCLUDE_EMPTY_BY_TICKER.get(ticker, True)}")
    log(f"Run audit_conditions.py first if INCLUDE_EMPTY not yet set for this ticker")

    out_file      = os.path.join(FEATURES_DIR, f"{ticker}_features.parquet")
    error_log     = os.path.join(REPORTS_DIR,  f"{ticker}_pipeline_errors.log")

    if os.path.exists(out_file):
        log(f"Already exists — delete to rerun: {out_file}")
        return

    quote_files = sorted(glob.glob(
        os.path.join(RAW_DIR, f"{ticker}_quotes_*.parquet")
    ))
    dates = [
        os.path.basename(f).replace(f"{ticker}_quotes_","").replace(".parquet","")
        for f in quote_files
    ]

    log(f"Found {len(dates)} days to process")

    all_days   = []
    failed     = 0
    skipped    = 0

    for i, date in enumerate(dates):
        try:
            day_df = process_day(ticker, date)
            if day_df is not None and len(day_df) > 0:
                all_days.append(day_df)
                print(f"  [{i+1:3d}/{len(dates)}] {date} -> {len(day_df):,} rows", flush=True)
            else:
                skipped += 1
                print(f"  [{i+1:3d}/{len(dates)}] {date} -> skipped (insufficient data)", flush=True)
        except Exception as e:
            failed += 1
            log_error(error_log, date, e)
            print(f"  [{i+1:3d}/{len(dates)}] {date} -> ERROR: {type(e).__name__}: {e}", flush=True)

    if all_days:
        final = pd.concat(all_days, ignore_index=True)
        final.to_parquet(out_file, compression="zstd", index=False)
        size_mb = os.path.getsize(out_file) / 1e6

        log(f"SAVED: {out_file}")
        log(f"Rows:    {len(final):,}")
        log(f"Size:    {size_mb:.1f} MB")
        log(f"Failed:  {failed} days — see {error_log}")
        log(f"Skipped: {skipped} days")
        log(f"Range:   {final['date'].min()} -> {final['date'].max()}")
        log(f"Columns: {final.columns.tolist()}")

        log("Feature means (sanity check):")
        numeric = final.select_dtypes(include=[np.number])
        print(numeric.mean().round(6).to_string())

        # Save summary stats to reports
        summary_path = os.path.join(REPORTS_DIR, f"{ticker}_feature_summary.csv")
        numeric.describe().to_csv(summary_path)
        log(f"Summary stats saved: {summary_path}")
    else:
        log(f"No data processed for {ticker}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 pipeline_features.py SPY")
        sys.exit(1)
    main(sys.argv[1].upper())
