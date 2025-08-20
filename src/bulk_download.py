#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bulk OHLCV downloader for BTC/USDT on Binance.
Downloads 5-minute bars from 2024-12-01 to 2025-08-19,
cleans + validates, and saves to data/raw/btcusdt_5m.csv
"""

import time
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import ccxt, requests, certifi

from data_cleaning import clean_ohlcv, validate_ohlcv

# Requests session (with certifi for SSL)
session = requests.Session()
session.verify = certifi.where()

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def bulk_download_binance(pair: str,
                          interval: str,
                          start: datetime,
                          end: datetime,
                          file_path: Path,
                          exchange_cfg: dict) -> pd.DataFrame:
    """
    Download full OHLCV history for given pair & timeframe between start and end.
    Saves a single cleaned CSV.
    """
    symbol    = pair                # "BTC/USDT"
    timeframe = f"{interval}m"      # "5m"
    limit     = 1500                # Binance max rows per fetch

    exchange  = ccxt.binance({
        "enableRateLimit": True,
        "session": session,
    })
    exchange.session.verify = certifi.where()

    all_batches = []

    since = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    logger.info("Starting bulk download: %s %s (%s → %s)",
                symbol, timeframe, start, end)

    while since < end_ms:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except Exception as e:
            logger.error("Fetch failed at %s: %s", pd.to_datetime(since, unit="ms"), e)
            break

        if not batch:
            logger.warning("No bars returned at since=%s", since)
            break

        df_new = pd.DataFrame(
            batch, columns=["time","open","high","low","close","volume"]
        )
        df_new["time"] = pd.to_datetime(df_new["time"], unit="ms")
        df_new.set_index("time", inplace=True)

        # Force numeric types (Binance returns floats, but be safe)
        for col in ["open","high","low","close","volume"]:
            df_new[col] = pd.to_numeric(df_new[col], errors="coerce")

        first, last = df_new.index[0], df_new.index[-1]
        all_batches.append(df_new)

        logger.info("Fetched %s → %s (%d rows)", first, last, len(df_new))

        since = int(last.timestamp() * 1000) + 1

        # Avoid hitting API rate limits
        time.sleep(0.8)

    if not all_batches:
        raise RuntimeError("No data fetched at all.")

    # Concatenate all batches
    df = pd.concat(all_batches).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Clean + validate
    df = clean_ohlcv(df, exchange_cfg)
    validate_ohlcv(df)

    # Save
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path)

    logger.info("Saved %d rows to %s", len(df), file_path)
    return df


if __name__ == "__main__":
    # Config
    start = datetime(2024, 12, 1)
    end   = datetime(2025, 8, 19)
    output_path = Path("data/raw/btcusdt_5m.csv")

    cfg = {"retries": 3, "retry_backoff": 5}

    bulk_download_binance(
        pair="BTC/USDT",
        interval="5",
        start=start,
        end=end,
        file_path=output_path,
        exchange_cfg=cfg,
    )
