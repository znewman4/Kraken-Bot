#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 10:42:13 2025

@author: zachnewman
"""

# src/data_loading.py

import os
import pandas as pd
import krakenex
from dotenv import load_dotenv

load_dotenv()

def get_kraken_client():
    """Initialize and return Kraken API client."""
    k = krakenex.API()
    k.key = os.getenv("KRAKEN_API_KEY")
    k.secret = os.getenv("KRAKEN_API_SECRET")
    return k

def fetch_ohlcv_kraken(pair='XBTUSD', interval=1):
    """
    Fetch OHLCV data from Kraken.

    Args:
        pair (str): Trading pair, e.g., 'XBTUSD'
        interval (int): Candle interval in minutes (1, 5, 15, 30, 60, etc.)

    Returns:
        pd.DataFrame: OHLCV data
    """
    k = get_kraken_client()
    response = k.query_public('OHLC', {'pair': pair, 'interval': interval})

    if response.get('error'):
        raise Exception(f"Kraken API error: {response['error']}")

    key = list(response['result'].keys())[0]  # Usually 'XXBTZUSD'
    ohlc = response['result'][key]
    df = pd.DataFrame(ohlc, columns=[
        'time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
    ])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Convert numeric columns from string to float
    numeric_cols = ['open', 'high', 'low', 'close', 'vwap', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df

def save_to_csv(df, filepath):
    """Save DataFrame to CSV."""
    df.to_csv(filepath)
