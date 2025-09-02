# src/backtesting/feeds.py
import backtrader as bt

class EngineeredData(bt.feeds.PandasData):
    lines = (
        'ema_10','sma_10','rsi_14',
        'bbl_20_2_0','bbm_20_2_0','bbu_20_2_0',
        'macd_12_26_9','macdh_12_26_9','macds_12_26_9',  # put all three here
        'volatility_5','log_return','log_return_1',
        'count','vwap','log_return_5','bbb_20_2_0','bbp_20_2_0',
    )
    params = (
        ('datetime', None),
        ('open', 'open'), ('high', 'high'), ('low', 'low'),
        ('close', 'close'), ('volume', 'volume'), ('openinterest', -1),

        ('ema_10', -1), ('sma_10', -1), ('rsi_14', -1),

        # Bollinger: dotted CSV → underscored line
        ('bbl_20_2_0', 'BBL_20_2.0'),
        ('bbm_20_2_0', 'BBM_20_2.0'),
        ('bbu_20_2_0', 'BBU_20_2.0'),
        ('bbb_20_2_0', 'BBB_20_2.0'),
        ('bbp_20_2_0', 'BBP_20_2.0'),

        # MACD: camel-case CSV → underscored line
        ('macd_12_26_9',  'MACD_12_26_9'),
        ('macdh_12_26_9', 'MACDh_12_26_9'),
        ('macds_12_26_9', 'MACDs_12_26_9'),

        ('volatility_5', -1), ('log_return', -1), ('log_return_1', -1),
        ('count', -1), ('vwap', -1), ('log_return_5', -1),
    )


class PrecomputedData(EngineeredData):
    lines = (
        'ema_10','sma_10','rsi_14',
        'bbl_20_2_0','bbm_20_2_0','bbu_20_2_0',
        'macd_12_26_9','macdh_12_26_9','macds_12_26_9',
        'volatility_5','log_return','log_return_1',
        'count','vwap','log_return_5','bbb_20_2_0','bbp_20_2_0',
        'exp_return',   # 👈 added here
    )

    params = (
        ('datetime', None),
        ('open', 'open'), ('high', 'high'), ('low', 'low'),
        ('close', 'close'), ('volume', 'volume'), ('openinterest', -1),

        ('ema_10', -1), ('sma_10', -1), ('rsi_14', -1),

        # Bollinger — underscore names for parquet
        ('bbl_20_2_0', 'bbl_20_2_0'),
        ('bbm_20_2_0', 'bbm_20_2_0'),
        ('bbu_20_2_0', 'bbu_20_2_0'),
        ('bbb_20_2_0', 'bbb_20_2_0'),
        ('bbp_20_2_0', 'bbp_20_2_0'),

        # MACD — underscore names for parquet
        ('macd_12_26_9',  'macd_12_26_9'),
        ('macdh_12_26_9', 'macdh_12_26_9'),
        ('macds_12_26_9', 'macds_12_26_9'),

        ('volatility_5', -1), ('log_return', -1), ('log_return_1', -1),
        ('count', -1), ('vwap', -1), ('log_return_5', -1),

        ('exp_return', -1),   # 👈 maps to itself
    )