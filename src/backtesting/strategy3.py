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
        # ——— Configuration —————————————————————————————————————————————
        self.cfg       = self.params.config
        self.tl_cfg    = self.cfg['trading_logic']
        self.model_cfg = self.cfg['model']

        # ——— Load each horizon model and its trained feature list ————————
        self.models              = {}
        self.model_feature_names = {}
        for h, path in self.tl_cfg['model_paths'].items():
            m = xgb.XGBRegressor()
            m.load_model(path)
            self.models[h] = m
            self.model_feature_names[h] = m.get_booster().feature_names

        # ——— Load the ordered list of features from disk —————————————————
        with open(self.model_cfg['features'], 'r') as f:
            self.feature_names = json.load(f)

        # ——— Metrics & buffers —————————————————————————————————————
        self.closes        = []
        self.ret_buffer    = []
        self.exp_returns   = []
        self.edge_norms    = []
        self.thresholds    = []
        self.signals       = []
        self.positions_log = []
        self.pnls          = []
        self.sig_buf       = deque(maxlen=self.tl_cfg.get('persistence', 3))

        # ——— Indicators for stops & features —————————————————————————
        atr_period = self.tl_cfg.get('atr_period', 14)
        self.atr    = bt.indicators.ATR(self.data, period=atr_period)
        self.ema_10 = bt.indicators.EMA(self.data.close, period=10)
        self.sma_10 = bt.indicators.SMA(self.data.close, period=10)
        self.rsi_14 = bt.indicators.RSI(self.data.close, period=14)
        macd_h      = MACDHisto(self.data.close, period_me1=12, period_me2=26, period_signal=9)
        self.macd_line, self.macd_sig, self.macd_hist = macd_h.macd, macd_h.signal, macd_h.histo
        bb = bt.indicators.BollingerBands(self.data.close, period=20, devfactor=2.0)
        self.bb_top, self.bb_mid, self.bb_bot = bb.top, bb.mid, bb.bot

        # ——— Rolling buffers sized for largest horizon + 1 ——————————————
        max_h      = max(self.tl_cfg['model_paths'].keys())
        self.log_ret   = deque(maxlen=max_h + 1)
        self.tp_buffer = deque(maxlen=self.tl_cfg.get('vol_window', 20))
        self.vol_buf   = deque(maxlen=self.tl_cfg.get('vol_window', 20))

        # ——— Placeholder for bracket orders ————————————————————————
        self.bracket_orders = []

        # ——— Log stop/tp multipliers —————————————————————————————
        sm = self.tl_cfg.get('stop_loss_atr_mult', 1.5)
        tm = self.tl_cfg.get('take_profit_atr_mult', 2.0)
        self.print_log(f"Using stop_mult={sm}, tp_mult={tm}")

    def print_log(self, txt):
        dt = datetime.utcnow()
        print(f"{dt.isoformat()} - {txt}")

    def log(self, txt):
        try:
            dt = self.datas[0].datetime.datetime(0)
        except:
            dt = datetime.utcnow()
        print(f"{dt.isoformat()} - {txt}")

    def notify_order(self, order):
        if order.status == bt.Order.Completed:
            act = 'BUY' if order.isbuy() else 'SELL'
            self.log(f"{act} EXECUTED: Price={order.executed.price:.2f}, Size={order.executed.size:.6f}")
        elif order.status in (bt.Order.Canceled, bt.Order.Margin, bt.Order.Rejected):
            status = order.Status[order.status] if hasattr(order, 'Status') else order.status
            self.log(f"⚠️ Order {status}")

    def next(self):
        # Grab current bar datetime
        dt = self.datas[0].datetime.datetime(0)

        # Warm-up for indicators
        if len(self) < 26:
            return

        # Update rolling buffers
        self.log_ret.append(np.log(self.data.close[0] / self.data.close[-1]))
        tp = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3
        self.tp_buffer.append(tp)
        self.vol_buf.append(self.data.volume[0])

        # Compute BB metrics
        top, mid, bot = float(self.bb_top[0]), float(self.bb_mid[0]), float(self.bb_bot[0])
        bandwidth = (top - bot) / mid * 100 if mid else 0.0
        percent   = (self.data.close[0] - bot) / (top - bot) if top != bot else 0.0

        # Build feature dict
        row = {}
        for feat in self.feature_names:
            f = feat.lower()
            if   f == 'open':   val = self.data.open[0]
            elif f == 'high':   val = self.data.high[0]
            elif f == 'low':    val = self.data.low[0]
            elif f == 'close':  val = self.data.close[0]
            elif f == 'volume': val = self.data.volume[0]
            elif f == 'vwap':   val = sum(p*v for p,v in zip(self.tp_buffer, self.vol_buf)) / sum(self.vol_buf)
            elif f == 'count':  val = len(self) - 1
            elif f == 'ema_10': val = float(self.ema_10[0])
            elif f == 'sma_10': val = float(self.sma_10[0])
            elif f == 'rsi_14': val = float(self.rsi_14[0])
            elif f in ('macd_12_26_9',):   val = float(self.macd_line[0])
            elif f in ('macdh_12_26_9',):  val = float(self.macd_hist[0])
            elif f in ('macds_12_26_9',):  val = float(self.macd_sig[0])
            elif f == 'bbl_20_2.0':        val = bot
            elif f == 'bbm_20_2.0':        val = mid
            elif f == 'bbu_20_2.0':        val = top
            elif f == 'bbb_20_2.0':        val = bandwidth
            elif f == 'bbp_20_2.0':        val = percent
            elif f.startswith('log_return_'):
                h = int(f.split('_')[-1])
                seq = list(self.log_ret)[-(h+1):-1] if len(self.log_ret) >= h+1 else []
                val = float(sum(seq)) if seq else 0.0
            elif f.startswith('volatility_'):
                h = int(f.split('_')[-1])
                seq = list(self.log_ret)[-(h+1):-1] if len(self.log_ret) >= h+1 else []
                val = float(np.std(seq, ddof=0)) if seq else 0.0
            else:
                val = getattr(self.data, f)[0]
            row[feat] = float(val)

        # Predict & average
        row_df = pd.DataFrame([row])
        preds  = []
        for h, m in self.models.items():
            cols = self.model_feature_names[h]
            Xh   = row_df.reindex(columns=cols)
            preds.append(m.predict(Xh)[0])
        exp_r = float(np.mean(preds))

        # Rolling realized-volatility
        if self.closes:
            self.ret_buffer.append((self.data.close[0] - self.closes[-1]) / self.closes[-1])
        self.closes.append(self.data.close[0])
        if len(self.ret_buffer) >= self.tl_cfg['vol_window']:
            vol = pd.Series(self.ret_buffer[-self.tl_cfg['vol_window']:]).std()
        else:
            vol = np.nan

        # Edge, threshold, signal
        if vol and not np.isnan(vol):
            edge = exp_r / vol
            thr  = (self.tl_cfg['fee_rate'] * self.tl_cfg.get('threshold_mult', 1.0)) / vol
        else:
            edge, thr = 0.0, float('inf')
        sig = 1 if edge >= thr else -1 if edge <= -thr else 0
        self.sig_buf.append(sig)

        # Position factor & record metrics
        conf     = abs(edge)
        max_conf = max(self.edge_norms) if self.edge_norms else 1e-8
        pos_fac  = (conf / max(max_conf, 1e-8)) * self.tl_cfg['max_position'] * sig

        self.exp_returns.append(exp_r)
        self.edge_norms.append(edge)
        self.thresholds.append(thr)
        self.signals.append(sig)
        self.positions_log.append(pos_fac)
        self.pnls.append(self.position.size * exp_r * self.data.close[0])

        # Debug edge vs threshold
        self.log(f"exp_r={exp_r:.4f}, edge={edge:.4f}, thr={thr:.4f}, pos_fac={pos_fac:.4f}")

        # Entry gating: allow both long and short
        if pos_fac == 0 or not any(s != 0 for s in self.sig_buf):
            return

        # Position sizing
        cash  = self.broker.getcash()
        price = self.data.close[0]
        size  = min((cash * abs(pos_fac)) / price, cash / price)

        # Place bracket orders
        stop_mult = self.tl_cfg['stop_loss_atr_mult']
        tp_mult   = self.tl_cfg['take_profit_atr_mult']
        stop_px   = price - stop_mult * self.atr[0] if sig > 0 else price + stop_mult * self.atr[0]
        limit_px  = price + tp_mult   * self.atr[0] if sig > 0 else price - tp_mult   * self.atr[0]

        self.log(f"Placing {'BUY' if sig>0 else 'SELL'} bracket: size={size:.6f}, stop_mult={stop_mult}, tp_mult={tp_mult}")
        if sig > 0:
            entry, stop, limit = self.buy_bracket(size=size, price=price, stopprice=stop_px, limitprice=limit_px)
        else:
            entry, stop, limit = self.sell_bracket(size=size, price=price, stopprice=stop_px, limitprice=limit_px)
        self.bracket_orders = [entry, stop, limit]

        # Periodic logging
        if dt.minute % 15 == 0:
            self.log(f"Signal={sig}, Edge={edge:.3f}, Thresh={thr:.3f}")

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
