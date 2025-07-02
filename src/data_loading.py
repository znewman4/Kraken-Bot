# src/data_loading.py

import time
from pathlib import Path

import pandas as pd
import ccxt  # pip install ccxt

def append_new_ohlcv(pair: str,
                     interval: int,
                     file_path: Path) -> pd.DataFrame:
    """
    1) Load existing CSV of raw OHLCV.
    2) Fetch up to 720 bars since the last timestamp + 1ms.
    3) Append, dedupe, save, and return the updated DataFrame.
    """
    # ensure parent folder exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # prepare CCXT
    symbol    = pair.replace("XBT", "BTC").replace("USD", "/USD")  # "BTC/USD"
    timeframe = f"{interval}m"
    limit     = 720
    exchange  = ccxt.kraken({ "enableRateLimit": True })

    # 1) load existing data
    if file_path.exists():
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        last_ms = int(df.index.max().timestamp() * 1000)
        since   = last_ms + 1
        print(f"🔎 Fetching bars since {df.index.max()} (+1 ms)")
    else:
        raise FileNotFoundError(f"No CSV at {file_path}; please seed with some data first")

    # 2) fetch one batch of new bars
    batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    if not batch:
        print("ℹ️ No new bars returned.")
        return df

    # 3) turn into DataFrame
    df_new = pd.DataFrame(batch, columns=["time","open","high","low","close","volume"])
    df_new["time"] = pd.to_datetime(df_new["time"], unit="ms")
    df_new.set_index("time", inplace=True)

    # append, dedupe, save
    df = pd.concat([df, df_new]).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.to_csv(file_path)
    print(f"💾 Appended {len(df_new)} rows up to {df_new.index[-1]} (total {len(df)} rows)")

    return df
