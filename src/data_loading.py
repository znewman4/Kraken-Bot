#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 10:42:13 2025

@author: zachnewman
"""

# src/data_loading.py

import os
import time
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

def fetch_ohlcv_kraken(pair='XBTUSD', interval=5, since=None):

    k = get_kraken_client()
    # Base params
    params = {'pair': pair, 'interval': interval}
    # Only include 'since' if provided (Kraken expects milliseconds)
    if since is not None:
        params['since'] = since

    response = k.query_public('OHLC', params)
    if response.get('error'):
        raise Exception(f"Kraken API error: {response['error']}")

    key = list(response['result'].keys())[0]
    ohlc = response['result'][key]
    df = pd.DataFrame(
        ohlc,
        columns=['time','open','high','low','close','vwap','volume','count']
    )
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    numeric_cols = ['open','high','low','close','vwap','volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df

def fetch_ohlcv_kraken_paginated(pair='XBTUSD', interval=5, since=None):
    """
    Fetch all OHLCV bars from `since` to now by paging through Kraken's limit.
    Returns a single DataFrame (possibly empty).
    """
    all_batches = []
    fetch_since = since

    while True:
        batch = fetch_ohlcv_kraken(pair=pair, interval=5, since=fetch_since)
        if batch.empty:
            break

        all_batches.append(batch)

        # advance to just after the last bar
        last_ts = batch.index[-1]
        fetch_since = int(last_ts.timestamp() * 1000) + interval * 60 * 1000

        time.sleep(1)  # avoid hammering the API

    if not all_batches:
        return pd.DataFrame()  # no new data

    df_all = pd.concat(all_batches)
    # dedupe and sort just in case of overlap
    df_all = df_all[~df_all.index.duplicated(keep='first')].sort_index()
    return df_all


def save_to_csv(df, filepath):
    """Save DataFrame to CSV."""
    df.to_csv(filepath)
