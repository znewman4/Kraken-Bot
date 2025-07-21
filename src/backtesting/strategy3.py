# src/backtesting/strategy3.py
print("Loading Strategy 3...")

import backtrader as bt
from backtrader.indicators import MACDHisto
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from collections import deque
from datetime import datetime


class KrakenStrategy(bt.Strategy):
    params = (('config', None),)

    def __init__(self):
        # Configuration
        self.cfg       = self.params.config
        self.tl_cfg    = self.cfg['trading_logic']
        self.model_cfg = self.cfg['model']

        # Load models
        self.models = {}
        for horizon, path in self.tl_cfg['model_paths'].items():
            model = xgb.XGBRegressor()
            model.load_model(path)
            self.models[horizon] = model

        # Load feature names
        with open(self.model_cfg['features'], 'r') as f:
            self.feature_names = json.load(f)
        print("▶️ Features loaded:", self.feature_names)

        # Metrics history
        self.closes        = []
        self.exp_returns   = []
        self.edge_norms    = []
        self.thresholds    = []
        self.signals       = []
        self.positions_log = []  # renamed to avoid conflict
        self.pnls          = []

        # Persistence for signals
        persistence = self.tl_cfg.get('persistence', 3)
        self.sig_buf = deque(maxlen=persistence)

        # ATR indicator
        atr_p = self.tl_cfg.get('atr_period', 14)
        self.atr = bt.indicators.ATR(self.data, period=atr_p)

        # On-the-fly indicators
        self.ema_10 = bt.indicators.EMA(self.data.close, period=10)
        self.sma_10 = bt.indicators.SMA(self.data.close, period=10)
        self.rsi_14 = bt.indicators.RSI(self.data.close, period=14)

        macd_h = MACDHisto(
            self.data.close,
            period_me1=12,
            period_me2=26,
            period_signal=9
        )
        self.macd_line = macd_h.macd
        self.macd_sig  = macd_h.signal
        self.macd_hist = macd_h.histo

        bb = bt.indicators.BollingerBands(
            self.data.close,
            period=20,
            devfactor=2.0
        )
        self.bb_top = bb.top
        self.bb_mid = bb.mid
        self.bb_bot = bb.bot

        # Rolling buffers
        self.log_ret   = deque(maxlen=6)
        self.tp_buffer = deque(maxlen=self.tl_cfg.get('vol_window', 20))
        self.vol_buf   = deque(maxlen=self.tl_cfg.get('vol_window', 20))

    def next(self):
        # Warm-up: need enough bars for slowest indicator
        if len(self) < 26:
            return

        # Update log-return
        lr = np.log(self.data.close[0] / self.data.close[-1])
        self.log_ret.append(lr)

        # Update VWAP buffers
        tp = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3
        self.tp_buffer.append(tp)
        self.vol_buf.append(self.data.volume[0])

        # Compute BB bandwidth & percent
        top       = float(self.bb_top[0])
        mid       = float(self.bb_mid[0])
        bot       = float(self.bb_bot[0])
        bandwidth = (top - bot) / mid * 100 if mid else 0.0
        percent   = (self.data.close[0] - bot) / (top - bot) if (top - bot) else 0.0

        # Build feature vector
        row = {}
        for feat in self.feature_names:
            f = feat.lower()
            if   f == 'open':       val = self.data.open[0]
            elif f == 'high':       val = self.data.high[0]
            elif f == 'low':        val = self.data.low[0]
            elif f == 'close':      val = self.data.close[0]
            elif f == 'volume':     val = self.data.volume[0]
            elif f == 'vwap':       val = sum(p*v for p,v in zip(self.tp_buffer, self.vol_buf)) / sum(self.vol_buf)
            elif f == 'count':      val = len(self) - 1
            elif f == 'ema_10':     val = float(self.ema_10[0])
            elif f == 'sma_10':     val = float(self.sma_10[0])
            elif f == 'rsi_14':     val = float(self.rsi_14[0])
            elif f in ('macd_12_26_9',):  val = float(self.macd_line[0])
            elif f in ('macdh_12_26_9',): val = float(self.macd_hist[0])
            elif f in ('macds_12_26_9',): val = float(self.macd_sig[0])
            elif f == 'bbl_20_2.0':       val = bot
            elif f == 'bbm_20_2.0':       val = mid
            elif f == 'bbu_20_2.0':       val = top
            elif f == 'bbb_20_2.0':       val = bandwidth
            elif f == 'bbp_20_2.0':       val = percent
            elif f == 'log_return':       val = self.log_ret[-1]
            elif f == 'log_return_1':     val = self.log_ret[-2]
            elif f == 'log_return_5':     val = sum(list(self.log_ret)[-6:-1])
            elif f == 'volatility_5':     val = float(np.std(list(self.log_ret)[-6:-1], ddof=0))
            else:                        val = getattr(self.data, f)[0]
            row[feat] = float(val)

        # Predict
        df    = pd.DataFrame([row])
        preds = [m.predict(df)[0] for m in self.models.values()]
        exp_r = float(np.mean(preds))

        # ... rest of your logic unchanged ...

    def get_metrics(self):
        return pd.DataFrame({
            'close':      self.closes,
            'exp_return': self.exp_returns,
            'edge_norm':  self.edge_norms,
            'threshold':  self.thresholds,
            'signal':     self.signals,
            'position':   self.positions_log,
            'pnl':        self.pnls
        })
