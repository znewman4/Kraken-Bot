# src/data_loading.py

import time
import logging
from pathlib import Path

import requests
import certifi
import pandas as pd
import ccxt

# ensure requests uses certifi bundle
session = requests.Session()
session.verify = certifi.where()

logger = logging.getLogger(__name__)

def fetch_ohlcv_with_retry(exchange, symbol, timeframe, since, limit, exchange_cfg):
    retries = exchange_cfg.get("retries", 3)
    backoff = exchange_cfg.get("retry_backoff", 5)
    for attempt in range(1, retries + 1):
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            logger.debug(
                "Fetched %d bars for %s/%s on attempt %d/%d",
                len(bars), symbol, timeframe, attempt, retries
            )
            return bars
        except Exception as e:
            if attempt < retries:
                logger.warning(
                    "fetch_ohlcv failed attempt %d/%d: %s – retrying in %ds…",
                    attempt, retries, e, backoff, exc_info=True
                )
                time.sleep(backoff)
            else:
                logger.error(
                    "All %d attempts to fetch_ohlcv failed for %s/%s: %s",
                    retries, symbol, timeframe, e, exc_info=True
                )
                raise

def append_new_ohlcv(
    pair: str,
    interval: int,
    file_path: Path,
    exchange_cfg: dict,
    start_date: str = None
) -> pd.DataFrame:
    """
    1) If CSV exists, load and fetch since last timestamp +1ms.
    2) If missing and start_date given, seed full history from start_date.
       If missing and no start_date, fetch only one batch.
    3) Append, dedupe, save, and return DataFrame.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # format symbol & timeframe
    symbol = pair.replace("XBT", "BTC").replace("USD", "/USD")
    timeframe = f"{interval}m"
    limit = 720

    exchange = ccxt.kraken({
        "enableRateLimit": True,
        "session": session,
    })
    exchange.session.verify = certifi.where()

    # 1) Load or seed
    if file_path.exists():
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        last_ts = df.index.max()
        since = int(last_ts.timestamp() * 1000) + 1
        logger.info("Existing CSV found; fetching from %s (+1 ms)", last_ts)
    else:
        df = pd.DataFrame()
        if start_date:
            since = int(pd.to_datetime(start_date).timestamp() * 1000)
            logger.info("No CSV found; seeding full history from %s", start_date)
        else:
            since = None
            logger.info(
                "No CSV found and no start_date; fetching only the last %d bars",
                limit
            )

    # 2) Fetch in batches until empty
    all_batches = []
    while True:
        batch = fetch_ohlcv_with_retry(
            exchange, symbol, timeframe, since, limit, exchange_cfg
        )
        if not batch:
            break
        df_batch = pd.DataFrame(batch, columns=["time","open","high","low","close","volume"])
        df_batch["time"] = pd.to_datetime(df_batch["time"], unit="ms")
        df_batch.set_index("time", inplace=True)
        all_batches.append(df_batch)
        since = int(df_batch.index.max().timestamp() * 1000) + 1
        time.sleep(exchange_cfg.get("retry_backoff", 5))

    # 3) Assemble and write
    df_new = pd.concat(all_batches) if all_batches else pd.DataFrame()
    df = pd.concat([df, df_new]).sort_index() if not df.empty else df_new
    df = df[~df.index.duplicated(keep="first")]
    df.to_csv(file_path)

    logger.info(
        "Seeded/appended %d rows up to %s (total %d rows)",
        len(df_new),
        df.index.max() if not df.empty else "n/a",
        len(df),
    )

    return df
