import time
import logging
from pathlib import Path

import requests, certifi
import pandas as pd
import ccxt  # pip install ccxt

# set up requests to use certifi’s bundle
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
                    attempt, retries, e, backoff,
                    exc_info=True
                )
                time.sleep(backoff)
            else:
                logger.error(
                    "All %d attempts to fetch_ohlcv failed for %s/%s: %s",
                    retries, symbol, timeframe, e,
                    exc_info=True
                )
                raise

def append_new_ohlcv(pair: str,
                     interval: str,
                     file_path: Path,
                     exchange_cfg: dict) -> pd.DataFrame:
    """
    1) Load existing CSV of raw OHLCV.
    2) Fetch up to 720 bars since the last timestamp + 1ms.
    3) Append, dedupe, save, and return the updated DataFrame.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # set up your exchange
    symbol    = pair.replace("XBT", "BTC").replace("USD", "/USD")  # "BTC/USD"
    timeframe = f"{interval}m"
    limit     = 720
    exchange  = ccxt.kraken({
        "enableRateLimit": True,
        "session": session,
    })
    exchange.session.verify = certifi.where()

    # 1) load existing data
    if file_path.exists():
        df     = pd.read_csv(file_path, index_col=0, parse_dates=True)
        last_ts = df.index.max()
        since   = int(last_ts.timestamp() * 1000) + 1
        logger.info("Fetching bars since %s (+1ms)", last_ts)
    else:
        raise FileNotFoundError(f"No CSV at {file_path}; please seed with some data first")

    # 2) fetch with retry
    batch = fetch_ohlcv_with_retry(
        exchange,
        symbol,
        timeframe,
        since,
        limit,
        exchange_cfg
    )

    if not batch:
        logger.info("No new bars returned.")
        return df

    # 3) convert & append
    df_new = pd.DataFrame(batch, columns=["time","open","high","low","close","volume"])
    df_new["time"] = pd.to_datetime(df_new["time"], unit="ms")
    df_new.set_index("time", inplace=True)

    df = pd.concat([df, df_new]).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index.name = "time"
    df.to_csv(file_path)

    # — this is now valid syntax —
    logger.info(
        "Appended %d rows up to %s (total %d rows)",
        len(df_new),
        df_new.index[-1],
        len(df)
    )

    return df
