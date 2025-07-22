# src/backtesting/feeds.py

import backtrader as bt

class EngineeredData(bt.feeds.PandasData):
    """
    A Backtrader feed that includes all engineered feature columns
    (including multi-horizon returns and volatilities) from the CSV.
    """
    # Define all extra lines, including log_return_3 and volatility_3
    lines = (
        'ema_10', 'sma_10', 'rsi_14',
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        'MACDs_12_26_9',
        'log_return', 'log_return_1', 'log_return_3', 'log_return_5',
        'volatility_3', 'volatility_5',
        'count', 'MACD_12_26_9', 'MACDh_12_26_9',
        'BBB_20_2.0', 'BBP_20_2.0', 'vwap',
    )

    # Map each line name to the corresponding CSV column name
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),

        # Technical indicators
        ('ema_10', -1),
        ('sma_10', -1),
        ('rsi_14', -1),
        ('BBL_20_2.0', -1),
        ('BBM_20_2.0', -1),
        ('BBU_20_2.0', -1),
        ('MACDs_12_26_9', -1),

        # Log returns for multiple horizons
        ('log_return',   -1),
        ('log_return_1', -1),
        ('log_return_3', -1),
        ('log_return_5', -1),

        # Rolling volatilities for multiple horizons
        ('volatility_3', -1),
        ('volatility_5', -1),

        # Other engineered features
        ('count',         -1),
        ('MACD_12_26_9',  -1),
        ('MACDh_12_26_9', -1),
        ('BBB_20_2.0',    -1),
        ('BBP_20_2.0',    -1),
        ('vwap',          -1),
    )
