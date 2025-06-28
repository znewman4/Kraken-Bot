#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 21:04:58 2025

@author: zachnewman
"""

# src/data_cleaning.py
import pandas as pd

def clean_ohlcv(df):
    """
    Clean raw OHLCV DataFrame.

    Steps:
    - Ensure datetime index is sorted
    - Remove duplicate timestamps
    - Handle missing values (simple forward fill or drop)
    """
    df = df.copy()

    # Sort index by time
    df.sort_index(inplace=True)

    # Drop duplicate timestamps
    df = df[~df.index.duplicated(keep='first')]

    # Handle missing values
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    return df


def validate_ohlcv(df):
    """
    Ensure OHLCV DataFrame has required columns and proper types.
    """
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise TypeError("Index must be datetime.")

