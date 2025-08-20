# src/technical_engineering.py

import pandas as pd
import pandas_ta as ta
import numpy as np



def add_technical_indicators(df, features_cfg):
    """
    Add technical indicators to OHLCV DataFrame.
    Includes: EMA, RSI, MACD, Bollinger Bands

    Args:
        df (pd.DataFrame): Clean OHLCV data

    Returns:
        pd.DataFrame: Data with added features
    """


    # at the start of add_technical_indicators(...)
    for c in ["open","high","low","close","volume","vwap","count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.copy()

    # EMA and SMA
    df['ema_10'] = ta.ema(df['close'], length=10)
    df['sma_10'] = ta.sma(df['close'], length=10)

    # RSI
    df['rsi_14'] = ta.rsi(df['close'], length=14)

    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    for col in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']:
        if col in df.columns:
            df = df.drop(columns=col)
    df = df.join(macd)
 

    # Bollinger Bands
    bb = ta.bbands(df['close'], length=20, std=2)
    df = df.join(bb)

    vwap_windows = features_cfg.get("vwap_windows", [20, 96, 288])
    df = add_vwap_and_count_features(df, windows=vwap_windows)

    return df

def add_return_features(df, features_cfg):
    """
    Add lagged returns and rolling volatility features.

    Args:
        df (pd.DataFrame): OHLCV data

    Returns:
        pd.DataFrame: With return-based features
    """
    df = df.copy()

    # Log returns
    df['log_return'] = (df['close'] / df['close'].shift(1)).apply(lambda x: pd.NA if x <= 0 else np.log(x))

    # Lagged returns
    df['log_return_1'] = df['log_return'].shift(1)
    df['log_return_5'] = df['log_return'].rolling(window=5).sum().shift(1)

    # Rolling volatility
    df['volatility_5'] = df['log_return'].rolling(window=5).std().shift(1)

    return df


def add_vwap_and_count_features(df: pd.DataFrame,
                                windows=(20, 96, 288),
                                use_fallback_when_missing=True) -> pd.DataFrame:
    """
    Adds:
      vwap_{w}           : rolling VWAP over window w (uses provided 'vwap' if present, else fallback)
      close_to_vwap_{w}  : (close - vwap_{w}) / vwap_{w}
      vwap_slope_{w}     : pct change of vwap_{w} over w bars
      log_count          : log1p(count) (only if 'count' present)
      avg_trade_size     : volume / count (only if 'count' present)
      count_z_{w}        : z-score of count over window w (only if 'count' present)
    All engineered cols are shifted by 1 to avoid lookahead.
    """
    out = df.copy()

    # Make sure base cols numeric
    for c in ["open","high","low","close","volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Base "per-bar" vwap series to roll on:
    # If exchange gave 'vwap', start from that; otherwise build a proxy from typical price.
    if "vwap" in out.columns and not out["vwap"].isna().all():
        vwap_bar = pd.to_numeric(out["vwap"], errors="coerce")
    else:
        # Typical price approximates intra-bar vwap reasonably
        tp = (out["high"] + out["low"] + out["close"]) / 3.0
        vwap_bar = tp

    # Rolling VWAP (volume-weighted) per window
    for w in windows:
        vol_sum = out["volume"].rolling(w, min_periods=w).sum()
        val_sum = (vwap_bar * out["volume"]).rolling(w, min_periods=w).sum()
        vwap_w  = val_sum / vol_sum

        out[f"vwap_{w}"] = vwap_w
        out[f"close_to_vwap_{w}"] = (out["close"] - vwap_w) / vwap_w
        out[f"vwap_slope_{w}"] = vwap_w.pct_change(w)

    # COUNT-based features ONLY if count exists (Kraken)
    if "count" in out.columns and not out["count"].isna().all():
        out["count"] = pd.to_numeric(out["count"], errors="coerce")
        c = out["count"].replace({0: np.nan})

        out["log_count"] = np.log1p(out["count"])
        out["avg_trade_size"] = out["volume"] / c

        for w in windows:
            mu = out["count"].rolling(w, min_periods=w).mean()
            sd = out["count"].rolling(w, min_periods=w).std()
            out[f"count_z_{w}"] = (out["count"] - mu) / sd

    # No lookahead
    new_cols = [col for col in out.columns if col not in df.columns]
    out[new_cols] = out[new_cols].shift(1)

    return out
