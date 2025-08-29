# src/bulk_download.py
import argparse
import math
import time, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.append(str(Path(__file__).resolve().parents[1]))  


import certifi
import pandas as pd
import requests

from config_loader import load_config

BINANCE_REST = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000  # Binance max klines per request


def to_binance_symbol(unified_symbol: str) -> str:
    """
    Map 'BTC/USDT' -> 'BTCUSDT', 'ETH/USDT' -> 'ETHUSDT', etc.
    """
    return unified_symbol.replace("/", "").replace(":", "")


def to_binance_interval(minutes: int) -> str:
    """
    Map minutes -> Binance kline interval string.
    """
    m = int(minutes)
    table = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 120: "2h", 240: "4h"}
    if m in table:
        return table[m]
    if m % 60 == 0:
        return f"{m // 60}h"
    return f"{m}m"


def read_last_timestamp(csv_path: Path) -> datetime | None:
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, usecols=["time"], parse_dates=["time"])
        if df.empty:
            return None
        return df["time"].max().to_pydatetime().replace(tzinfo=timezone.utc)
    except Exception:
        return None


def ceil_to_next_interval(dt_utc: datetime, minutes: int) -> datetime:
    # move to next multiple of interval after dt_utc
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    ms = int((dt_utc - epoch).total_seconds() * 1000)
    step = minutes * 60 * 1000
    next_ms = ((ms // step) + 1) * step
    return epoch + timedelta(milliseconds=next_ms)


def fetch_binance_klines(symbol: str, interval: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    """
    Streams Binance klines [start_utc, end_utc) inclusive of start.
    Returns DataFrame with columns:
    time, open, high, low, close, volume, quote_volume, trades
    """
    session = requests.Session()
    session.verify = certifi.where()

    retry = Retry(
    total=5,                # up to 5 retries
    backoff_factor=0.5,     # 0.5, 1.0, 2.0, ...
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    out_rows: list[list] = []
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }
        try:
            r = session.get(BINANCE_REST, params=params, timeout=30)
            r.raise_for_status()
            klines = r.json()
        except requests.exceptions.ReadTimeout:
            time.sleep(1.0)
            continue


        if not klines:
            break

        for k in klines:
            # kline fields: [0] openTime, [1] open, [2] high, [3] low, [4] close,
            # [5] volume, [6] closeTime, [7] quoteVolume, [8] trades, ...
            open_time_ms = int(k[0])
            t = datetime.fromtimestamp(open_time_ms / 1000.0, tz=timezone.utc)
            open_ = float(k[1]); high = float(k[2]); low = float(k[3]); close = float(k[4])
            vol_base = float(k[5])
            quote_vol = float(k[7])
            trades = int(k[8])
            out_rows.append([t, open_, high, low, close, vol_base, quote_vol, trades])

        # paginate: next start = last closeTime + 1ms
        last_close_ms = int(klines[-1][6])
        next_start_ms = last_close_ms + 1
        if next_start_ms >= end_ms:
            break
        # safety: if Binance returns fewer than limit, small sleep to avoid bans
        if len(klines) < MAX_LIMIT:
            time.sleep(0.05)
        start_ms = next_start_ms

    if not out_rows:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "quote_volume", "trades"])

    df = pd.DataFrame(out_rows, columns=["time", "open", "high", "low", "close", "volume", "quote_volume", "trades"])
    # make time tz-naive to match your Kraken CSV formatting
    df["time"] = pd.to_datetime(df["time"]).dt.tz_convert(None)
    return df


def compute_vwap_and_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # True bar VWAP from Binance fields (matches Kraken semantics)
    # Handle rare zero-volume bars gracefully
    df["vwap"] = df.apply(
        lambda r: (r["quote_volume"] / r["volume"]) if r["volume"] > 0 else r["close"],
        axis=1,
    )
    # Rename trades -> count (Kraken term)
    df["count"] = df["trades"].astype(float)
    return df


def to_kraken_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce the exact column order & types your pipeline expects.
    Columns: time, open, high, low, close, vwap, volume, count
    """
    df = df[["time", "open", "high", "low", "close", "vwap", "volume", "count"]].copy()
    # type stability (floats for prices/volume/vwap/count)
    for col in ["open", "high", "low", "close", "vwap", "volume", "count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # format time as 'YYYY-MM-DD HH:MM:SS' (no timezone)
    df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def append_or_write_csv(df_new: pd.DataFrame, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        df_old = pd.read_csv(out_csv, parse_dates=["time"])
        # bring new to same dtype/format as old
        df_new["time"] = pd.to_datetime(df_new["time"])
        df_all = (
            pd.concat([df_old, df_new], ignore_index=True)
            .drop_duplicates(subset=["time"])
            .sort_values("time")
        )
        # write back in same format as your existing file
        df_all["time"] = df_all["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df_all.to_csv(out_csv, index=False)
    else:
        df_new.to_csv(out_csv, index=False)


def main():
    cfg = load_config("config.yml")
    data_cfg = cfg.get("data", {})
    ex_cfg = cfg.get("exchange", {})

    out_csv = Path(data_cfg.get("raw_data_path", "data/raw/btc_ohlcv_5min_raw.csv"))

    symbol_unified = ex_cfg.get("symbol", "BTC/USDT")   # <— use config, not hard-coded
    symbol_binance = to_binance_symbol(symbol_unified)

    minutes = int(ex_cfg.get("interval_minute", 5))
    interval = to_binance_interval(minutes)

    # Determine [start, end) window
    last_ts = read_last_timestamp(out_csv)

     # read config start/end dates (if provided)
    start_date_cfg = data_cfg.get("start_date")
    end_date_cfg   = data_cfg.get("end_date")
    start_cfg = pd.to_datetime(start_date_cfg).to_pydatetime().replace(tzinfo=timezone.utc) if start_date_cfg else None
    end_cfg   = pd.to_datetime(end_date_cfg).to_pydatetime().replace(tzinfo=timezone.utc) if end_date_cfg else None

    if last_ts is None:
        if start_cfg is None:
                raise ValueError("No existing CSV and no data.start_date set in config.yml")
        start_utc = start_cfg
        end_utc   = end_cfg or datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
    else:
        start_utc = ceil_to_next_interval(last_ts, minutes)
        end_utc   = end_cfg or datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)

    if start_utc >= end_utc:
        print("No new bars to fetch. ✅")
        return

    print(f"Fetching Binance klines for {symbol_binance} interval {interval}")
    print(f"Window: {start_utc.isoformat()}  →  {end_utc.isoformat()}")

    df = fetch_binance_klines(symbol_binance, interval, start_utc, end_utc)
    if df.empty:
        print("No data returned.")
        return

    df = compute_vwap_and_count(df)
    df = to_kraken_schema(df)

    # de-dup & order just in case
    df = df.drop_duplicates(subset=["time"]).sort_values("time")

    append_or_write_csv(df, out_csv)

    print(f"✅ Wrote/updated: {out_csv}  (+{len(df)} new rows)")


if __name__ == "__main__":
    main()
