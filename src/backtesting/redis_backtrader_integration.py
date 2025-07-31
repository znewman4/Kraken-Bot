
# redis_backtrader_integration.py
# ---------------------------------------
# Full end-to-end pipeline (backtest edition):
# 1) Producer reads historical OHLCV bars from CSV.
# 2) Computes ALL technical features via your `technical_engineering` module.
# 3) Computes lagged returns & rolling volatility features.
# 4) Uses preloaded XGBoost models to compute exp_return_<h> for each horizon.
# 5) Publishes raw + features + exp_returns into a Redis Stream.
# 6) Backtrader `RedisFeatureFeed` XREADs the same stream,
#    buffers messages, and serves them bar-by-bar to any strategy.
# 7) Strategy consumes precomputed `exp_return_<h>` or rebuilds
#    feature vectors for in-strategy inference.
import sys, os
# ensure repo root is on PYTHONPATH so config_loader & technical_engineering can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import redis
from collections import deque
import json
import pandas as pd
import numpy as np
import backtrader as bt
import threading
from datetime import datetime
from xgboost import XGBRegressor
from config_loader import load_config
import src.technical_engineering as te

# at top of redis_backtrader_integration.py, below imports
HISTORY = deque(maxlen=100)


# ------------------------------
# 1) Configuration & Model Loading
# ------------------------------
cfg = load_config('config.yml')
MODEL_PATHS = cfg['trading_logic']['model_paths']
STREAM_KEY = cfg['stream']['feature_stream']

# Redis client
redis_client = redis.Redis(
    host=cfg['stream']['host'],
    port=cfg['stream']['port'],
    db=cfg['stream']['db']
)

# Load XGB models and feature names
MODELS = {}
FEATURE_NAMES = {}
for h_str, path in MODEL_PATHS.items():
    h = int(h_str)
    model = XGBRegressor()
    model.load_model(path)
    MODELS[h] = model
    FEATURE_NAMES[h] = model.get_booster().feature_names

# ------------------------------
# 2) Producer: feature & signal computation
# ------------------------------


def publish_bar(raw_bar: dict):
    """
    raw_bar must include at least:
      {'time', 'open', 'high', 'low', 'close', 'volume'}

    Steps:
      1. Build a 1-row DataFrame from raw_bar.
      2. Compute technical indicators.
      3. Compute return-based features.
      4. Compute exp_return_<h> via each model.
      5. Push all fields to Redis stream.
    """
    # 1) Base DataFrame
    HISTORY.append(raw_bar)
    if len(HISTORY) < 50:      # or whatever your longest lookback is
        return                 # skip until you have enough bars
    df_hist = pd.DataFrame(HISTORY)
    # now safe to call add_technical_indicators(df_hist, …)

    # 2) Technical indicators
    df_feats = te.add_technical_indicators(df_hist, cfg['features'])
    df_feats = te.add_return_features(df_feats, cfg['features'])
    feats = df_feats.iloc[-1].to_dict()

    # models expect 'count' among the features
    feats['count'] = raw_bar.get('count', np.nan)

    # 4) Model predictions
    exp_rs = {}
    for h, model in MODELS.items():
        vec = {f: feats[f] for f in FEATURE_NAMES[h]}
        pred = model.predict(pd.DataFrame([vec]))[0]
        exp_rs[h] = float(pred)

    # 5) Prepare and publish message
    message = {**raw_bar, **feats}
    for h, val in exp_rs.items():
        message[f'exp_return_{h}'] = val
    fields = {k: json.dumps(v) for k, v in message.items()}
    #print(f"[DEBUG] XADD → stream: {STREAM_KEY}; sample fields: {list(message.items())[:3]}")
    entry_id = redis_client.xadd(STREAM_KEY, fields)
    #print(f"[DEBUG] XADD returned entry ID: {entry_id}")



def publish_from_csv(csv_path=None, limit=None):
    """
    Backtest ingestion: read OHLCV bars from CSV, publish each via publish_bar().
    Args:
      csv_path: path to your historical data CSV (default from config).
      limit:  max number of rows to process (for quick tests).
    """
    if csv_path is None:
        csv_path = cfg['data']['feature_data_path']

    df = pd.read_csv(csv_path, parse_dates=['time'])
    if limit:
        df = df.head(limit)

    for idx, row in df.iterrows():
        raw_bar = {
            'time':   row['time'].isoformat(),
            'open':   row['open'],
            'high':   row['high'],
            'low':    row['low'],
            'close':  row['close'],
            'volume': row['volume'],
            'count':  row.get('count', np.nan),   # ← ADD THIS LINE

        }
        publish_bar(raw_bar)

# ------------------------------
# 3) Backtrader Data Feed: RedisFeatureFeed
# ------------------------------
class RedisFeatureFeed(bt.feed.DataBase):
    """Consumes feature_stream, outputs bars + features + exp_return_<h>."""
    params = (
        ('host', cfg['stream']['host']),
        ('port', cfg['stream']['port']),
        ('db', cfg['stream']['db']),
        ('stream_key', STREAM_KEY),
    )

    def __init__(self):
        self.redis = redis.Redis(host=self.p.host, port=self.p.port, db=self.p.db)
        self.last_id = '0-0'
        self.buffer = []
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while True:
            resp = self.redis.xread({self.p.stream_key: self.last_id}, block=500, count=50)
            if not resp:
                continue
            for _, msgs in resp:
                for mid, data in msgs:
                    entry = {k.decode(): json.loads(v.decode()) for k, v in data.items()}
                    self.buffer.append(entry)
                    self.last_id = mid.decode()

    def _load(self):
        if not self.buffer:
            return None
        fb = self.buffer.pop(0)
        dt = bt.date2num(pd.to_datetime(fb['time']))

        vals = [dt]
        # OHLCV
        for fld in ['open','high','low','close','volume']:
            vals.append(fb[fld])
        # All features
        # derive feature list once
        df_tmp = te.add_return_features(
            te.add_technical_indicators(
                pd.DataFrame([fb]), cfg['features']),
            cfg['features'])
        for col in df_tmp.columns:
            if col in ['time','open','high','low','close','volume']:
                continue
            vals.append(fb.get(col, np.nan))
        # exp_return_<h>
        for h in sorted(MODELS.keys()):
            vals.append(fb.get(f'exp_return_{h}', np.nan))

        return vals

    @classmethod
    def getlinealiases(cls):
        aliases = {'datetime': 0}
        idx = 1
        for fld in ['open','high','low','close','volume']:
            aliases[fld] = idx; idx += 1
        # feature columns
        df_tmp = te.add_return_features(
            te.add_technical_indicators(
                pd.DataFrame([{ 'time':'','open':0,'high':0,'low':0,'close':0,'volume':1 }]),
                cfg['features']),
            cfg['features'])
        for col in df_tmp.columns:
            if col in ['time','open','high','low','close','volume']:
                continue
            aliases[col] = idx; idx += 1
        # exp_return_<h>
        for h in sorted(MODELS.keys()):
            aliases[f'exp_return_{h}'] = idx; idx += 1
        return aliases

