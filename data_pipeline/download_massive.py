import os
import time
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta

# =====================================================
# CONFIG
# =====================================================
API_KEY = os.environ["MASSIVE_API_KEY"]   # set MASSIVE_API_KEY in your shell; never commit keys

BASE_URL = "https://api.massive.com/v3"
OUT_DIR = "/home/kethanse66/quant-research-ofi/raw_parquet"

START_DATE = datetime(2024, 4, 27)
END_DATE   = datetime(2025, 4, 27)

TICKERS = ["SPY", "IWM", "XLE", "XLV", "TLT"]
ENDPOINTS = ["quotes", "trades"]

LIMIT = 50000

os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================
# HELPERS
# =====================================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def trading_days(start, end):
    cur = start
    days = []
    while cur < end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def day_file(ticker, endpoint, dt):
    d = dt.strftime("%Y-%m-%d")
    return f"{OUT_DIR}/{ticker}_{endpoint}_{d}.parquet"


def checkpoint_file():
    return f"{OUT_DIR}/download_checkpoint.txt"


# =====================================================
# SAFE REQUEST
# =====================================================
def get_json(url, params):
    while True:
        try:
            r = requests.get(url, params=params, timeout=60)

            if r.status_code == 200:
                return r.json()

            if r.status_code == 429:
                log("Rate limited. Sleep 30s")
                time.sleep(30)
                continue

            if r.status_code in (401, 403):
                raise Exception("Auth failed")

            log(f"HTTP {r.status_code}")
            time.sleep(5)

        except Exception as e:
            log(f"Retrying after error: {e}")
            time.sleep(5)


# =====================================================
# DOWNLOAD ONE DAY
# =====================================================
def download_day(ticker, endpoint, dt):
    outfile = day_file(ticker, endpoint, dt)

    # Skip if already exists
    if os.path.exists(outfile):
        log(f"SKIP existing {os.path.basename(outfile)}")
        return True

    date_str = dt.strftime("%Y-%m-%d")

    url = f"{BASE_URL}/{endpoint}/{ticker}"
    params = {
        "timestamp": date_str,
        "limit": LIMIT,
        "sort": "timestamp",
        "order": "asc",
        "apiKey": API_KEY
    }

    writer = None
    total_rows = 0
    page = 0

    while True:
        data = get_json(url, params)
        rows = data.get("results", [])

        if not rows:
            break

        df = pd.DataFrame(rows)

        # Filter exact day only
        ts_col = None
        for c in ["sip_timestamp", "participant_timestamp", "timestamp"]:
            if c in df.columns:
                ts_col = c
                break

        if ts_col:
            ts = pd.to_datetime(df[ts_col], unit="ns", errors="coerce")
            df = df[ts.dt.date == dt.date()]

        if len(df) > 0:
            table = pa.Table.from_pandas(df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    outfile,
                    table.schema,
                    compression="zstd"
                )

            writer.write_table(table)

            page += 1
            total_rows += len(df)

            log(
                f"{ticker} {endpoint} {date_str} | "
                f"page {page} | rows {len(df):,} | total {total_rows:,}"
            )

        next_url = data.get("next_url")

        if not next_url:
            break

        url = next_url
        params = {"apiKey": API_KEY}

        time.sleep(0.10)

    if writer:
        writer.close()

    if total_rows > 0:
        mb = os.path.getsize(outfile) / 1e6
        log(f"SAVED {os.path.basename(outfile)} | {total_rows:,} rows | {mb:.1f} MB")
        return True

    return False


# =====================================================
# MAIN
# =====================================================
def run():
    days = trading_days(START_DATE, END_DATE)

    for ticker in TICKERS:
        for endpoint in ENDPOINTS:

            log("=" * 70)
            log(f"START {ticker} {endpoint.upper()}")
            log("=" * 70)

            for dt in days:
                try:
                    download_day(ticker, endpoint, dt)
                except Exception as e:
                    log(f"FAILED {ticker} {endpoint} {dt.date()} : {e}")

                time.sleep(0.25)


if __name__ == "__main__":
    print("=" * 70)
    print("FINAL DAILY FILE DOWNLOADER")
    print("No overwrite bug | Low RAM | Safe files")
    print("=" * 70)

    run()

    print("ALL COMPLETE")
