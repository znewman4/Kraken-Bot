import backtrader as bt

class EngineeredData(bt.feeds.PandasData):
    lines = (
        'ema_10', 'sma_10', 'rsi_14',
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        'MACDs_12_26_9', 'volatility_5', 'log_return', 'log_return_1',
        'count', 'MACD_12_26_9', 'MACDh_12_26_9',
        'BBB_20_2.0', 'BBP_20_2.0', 'vwap', 'log_return_5',
    )
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
        ('ema_10', -1),
        ('sma_10', -1),
        ('rsi_14', -1),
        ('BBL_20_2.0', -1),
        ('BBM_20_2.0', -1),
        ('BBU_20_2.0', -1),
        ('MACDs_12_26_9', -1),
        ('volatility_5', -1),
        ('log_return', -1),
        ('log_return_1', -1),
        ('count', -1),
        ('MACD_12_26_9', -1),
        ('MACDh_12_26_9', -1),
        ('BBB_20_2.0', -1),
        ('BBP_20_2.0', -1),
        ('vwap', -1),
        ('log_return_5', -1),
    )
