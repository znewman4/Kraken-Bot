# src/feature_engineering.py

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
