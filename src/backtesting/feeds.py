# src/backtesting/feeds.py
import backtrader as bt



import re
import backtrader as bt

# src/backtesting/feeds.py
import re
import backtrader as bt

BASE_COLS = {'open','high','low','close','volume','openinterest'}

def _to_attr(name: str) -> str:
    # normalize to a valid python identifier used by Backtrader
    s = re.sub(r'\W+', '_', str(name))   # dots -> underscores, remove other symbols
    if re.match(r'^\d', s):
        s = '_' + s
    return s.lower()                     # LOWERCASE so it matches strategy expectation

def make_dynamic_pandasdata(df, keep_cols=None):
    cols = list(df.columns) if keep_cols is None else [c for c in df.columns if c in keep_cols]

    mappings = []
    for col in cols:
        if str(col).lower() in BASE_COLS:
            continue
        line_name = _to_attr(col)        # <<< normalized attr name
        mappings.append((line_name, col)) # map line -> original df column

    class DynamicEngineeredData(bt.feeds.PandasData):
        lines = tuple([ln for ln, _ in mappings])
        params = (
            ('datetime', None),
            ('open', 'open'),
            ('high', 'high'),
            ('low', 'low'),
            ('close', 'close'),
            ('volume', 'volume'),
            ('openinterest', -1),
        ) + tuple((ln, col) for ln, col in mappings)

    return DynamicEngineeredData









class EngineeredData(bt.feeds.PandasData):
    lines = (
        'ema_10', 'sma_10', 'rsi_14',
        'bbl_20_2_0', 'bbm_20_2_0', 'bbu_20_2_0',   # ← underscore versions
        'macds_12_26_9', 'volatility_5', 'log_return', 'log_return_1',
        'count', 'macd_12_26_9', 'macdh_12_26_9',
        'bbb_20_2_0', 'bbp_20_2_0',                # ← underscore versions
        'vwap', 'log_return_5',
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

        # map underscored lines to dotted CSV column names
        ('bbl_20_2_0', 'BBL_20_2.0'),
        ('bbm_20_2_0', 'BBM_20_2.0'),
        ('bbu_20_2_0', 'BBU_20_2.0'),
        ('bbb_20_2_0', 'BBB_20_2.0'),
        ('bbp_20_2_0', 'BBP_20_2.0'),

        ('macds_12_26_9', -1),
        ('volatility_5', -1),
        ('log_return', -1),
        ('log_return_1', -1),
        ('count', -1),
        ('macd_12_26_9', -1),
        ('macdh_12_26_9', -1),

        ('vwap', -1),
        ('log_return_5', -1),
    )


class PrecomputedData(EngineeredData):
    lines = ('exp_return',)   # add only the new one
    params = (('exp_return', -1),)