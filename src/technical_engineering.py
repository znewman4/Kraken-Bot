# src/technical_engineering.py
import pandas as pd
import pandas_ta as ta
import numpy as np

def add_technical_indicators(df, features_cfg):
    """
    Add technical indicators to OHLCV DataFrame.
    Now includes: EMA, SMA, RSI, MACD, Bollinger Bands, VWAP, ATR, and preserves tick count.
    """
    df = df.copy()

        # — Convert the time column to datetime and set as index —
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
    df = df.sort_index()


    # ─── Moving Averages ──────────────────────────────────────────────────────────
    df['ema_10'] = ta.ema(df['close'], length=10)
    df['sma_10'] = ta.sma(df['close'], length=10)

    # ─── Momentum ─────────────────────────────────────────────────────────────────
    df['rsi_14'] = ta.rsi(df['close'], length=14)

    # ─── Trend / Convergence ─────────────────────────────────────────────────────
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df = df.join(macd)  # yields MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9

    # ─── Volatility Bands ─────────────────────────────────────────────────────────
    bb = ta.bbands(df['close'], length=20, std=2)
    df = df.join(bb)    # BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0

    # ─── VWAP ────────────────────────────────────────────────────────────────────
    # Use the full-history VWAP, equivalent to your manual loop
    # (you can also pass a rolling length via features_cfg if you want)
    df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])

    # ─── ATR ─────────────────────────────────────────────────────────────────────
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # ─── Tick Count ──────────────────────────────────────────────────────────────
    # If your raw bars include 'count', it’s already in df; but fill any missing with 0
    if 'count' in df.columns:
        df['count'] = df['count'].fillna(0)

    return df

def add_return_features(df, features_cfg):
    """
    Add lagged returns and rolling volatility features.
    """
    df = df.copy()

    # Log returns (shifted so that the return at t is for close[t]/close[t-1])
    df['log_return']   = np.log(df['close'] / df['close'].shift(1)).where(lambda x: x>0)

    # Lagged and aggregated returns
    df['log_return_1'] = df['log_return'].shift(1)
    df['log_return_5'] = df['log_return'].rolling(window=5).sum().shift(1)

    # Rolling volatility (std of returns)
    df['volatility_5'] = df['log_return'].rolling(window=5).std().shift(1)

    return df
