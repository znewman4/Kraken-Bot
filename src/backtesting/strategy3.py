# src/backtesting/strategy3.py

print("Loading Strategy 3...")

import backtrader as bt
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from collections import deque
from datetime import datetime


class KrakenStrategy(bt.Strategy):
    params = (
        ('config', None),
    )

    def __init__(self):

        self.cfg           = self.params.config
        self.tl_cfg        = self.cfg['trading_logic']
        self.model_cfg     = self.cfg['model']
        self.models        = {}
        self.feature_names = []

        # Load XGBoost models
        for horizon, path in self.tl_cfg['model_paths'].items():
            m = xgb.XGBRegressor()
            m.load_model(path)
            self.models[horizon] = m

        # Load feature names
        with open(self.model_cfg['features'], 'r') as f:
            self.feature_names = json.load(f)


        # Data trackers
        self.closes        = []
        self.ret_buffer    = []
        self.exp_returns   = []
        self.edge_norms    = []
        self.thresholds    = []
        self.signals       = []
        self.positions_log = []
        self.pnls          = []

        # Signal persistence
        persistence = self.tl_cfg.get('persistence', 3)
        self.signal_buffer = deque(maxlen=persistence)

        # ATR for bracket stops
        atr_p = self.tl_cfg.get('atr_period', 14)
        self.atr = bt.indicators.ATR(self.data, period=atr_p)

        # ON-THE-FLY INDICATORS
        # SMA
        sma_w = 20
        for ind in self.cfg.get('features', {}).get('indicators', []):
            if 'sma' in ind:
                sma_w = ind['sma'].get('window', sma_w)
        self.sma = bt.indicators.SMA(self.data.close, period=sma_w)

        # RSI
        rsi_w = 14
        for ind in self.cfg.get('features', {}).get('indicators', []):
            if 'rsi' in ind:
                rsi_w = ind['rsi'].get('window', rsi_w)
        self.rsi = bt.indicators.RSI(self.data.close, period=rsi_w)

        # VWAP buffers (we’ll compute VWAP as TP*vol / vol over vol_window)
        self.vwap_period = self.tl_cfg.get('vol_window', 20)
        self.tp_buffer   = deque(maxlen=self.vwap_period)
        self.vol_buffer  = deque(maxlen=self.vwap_period)

        # bracket orders holder
        self.bracket_orders = []

        sm = self.tl_cfg.get('stop_loss_atr_mult', 1.5)
        tm = self.tl_cfg.get('take_profit_atr_mult', 2.0)
        self.print_log(f"Using stop_mult={sm}, tp_mult={tm}")

    def print_log(self, txt):
        """Logger for __init__, uses UTC."""
        dt = datetime.utcnow()
        print(f"{dt.isoformat()} - {txt}")

    def notify_trade(self, trade):
    # Only record once the trade is closed
        if trade.isclosed:
            pnl = trade.pnl        # gross PnL
            pnl_comm = trade.pnlcomm  # net PnL after commission
            # log it
            self.log(f"TRADE CLOSED  |  GROSS PnL={pnl:.2f}  NET PnL={pnl_comm:.2f}")
            # append to your pnls list for true realized PnL
            self.pnls.append(pnl_comm)


    def log(self, txt):
        """Main logger, falls back to UTC if no bar yet."""
        try:
            dt = self.datas[0].datetime.datetime(0)
        except Exception:
            dt = datetime.utcnow()
        print(f"{dt.isoformat()} - {txt}")

    def notify_order(self, order):
        status = order.Status[order.status] if hasattr(order, 'Status') else order.status
        if order.status == bt.Order.Completed:
            act = 'BUY' if order.isbuy() else 'SELL'
            self.log(f"{act} EXECUTED: Price={order.executed.price}, Size={order.executed.size}")
        elif order.status in (bt.Order.Canceled, bt.Order.Margin, bt.Order.Rejected):
            self.log(f"⚠️ Order {status}")

    def next(self):
        # — Update VWAP buffers first —
        tp = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3.0
        self.tp_buffer.append(tp)
        self.vol_buffer.append(self.data.volume[0])

        # 1) Build feature row
        row = {}
        for f in self.feature_names:
            lf = f.lower()
            if lf == 'sma':
                row[f] = float(self.sma[0])
            elif lf == 'rsi':
                row[f] = float(self.rsi[0])
            elif lf == 'vwap':
                if sum(self.vol_buffer) > 0:
                    vwap_val = sum(p * v for p, v in zip(self.tp_buffer, self.vol_buffer)) / sum(self.vol_buffer)
                else:
                    vwap_val = tp
                row[f] = float(vwap_val)
            else:
                # raw fields: open, high, low, close, volume, etc.
                row[f] = float(getattr(self.data, f)[0])

        df    = pd.DataFrame([row])
        preds = [m.predict(df)[0] for m in self.models.values()]
        exp_r = float(np.mean(preds))

        # 2) Rolling volatility
        if self.closes:
            r = (self.data.close[0] - self.closes[-1]) / self.closes[-1]
            self.ret_buffer.append(r)
        self.closes.append(self.data.close[0])
        if len(self.ret_buffer) >= self.tl_cfg['vol_window']:
            vol = pd.Series(self.ret_buffer[-self.tl_cfg['vol_window']:]).std()
        else:
            vol = np.nan

        # 3) Edge & threshold
        if not np.isnan(vol) and vol > 0:
            edge = exp_r / vol
            thr  = (self.tl_cfg['fee_rate'] * self.tl_cfg.get('threshold_mult', 1.0)) / vol
        else:
            edge, thr = 0.0, float('inf')

        # 4) Signal + persistence
        sig = 1 if edge >= thr else (-1 if edge <= -thr else 0)
        self.signal_buffer.append(sig)

        # 5) Position factor (raw)
        conf     = abs(edge)
        max_conf = max(self.edge_norms) if self.edge_norms else 1e-8
        pos_fac  = (conf / max(max_conf, 1e-8)) * self.tl_cfg['max_position'] * sig

        # 5b) Clamp into [-max_position, +max_position]
        mp = self.tl_cfg['max_position']
        pos_fac = max(min(pos_fac, mp), -mp)



       
        # 6) Record metrics
        self.exp_returns.append(exp_r)
        self.edge_norms.append(edge)
        self.thresholds.append(thr)
        self.signals.append(sig)
        self.positions_log.append(pos_fac)

        # 7) Entry gating
        dt = self.datas[0].datetime.datetime(0)
        # 1) Only trade in your allowed hours
        if not (8 <= dt.hour <= 20):
            return

        # 2) Don’t trade if there’s no signal
        if sig == 0:
            return

        # 3) Require N‐bar persistence of the same sign
        if len(self.signal_buffer) >= self.signal_buffer.maxlen:
            if not all(s == sig for s in self.signal_buffer):
                return

        # 8) Sizing
        size_req   = (self.broker.getcash() * abs(pos_fac)) / self.data.close[0]
        max_afford = self.broker.getcash() / self.data.close[0]
        size       = min(size_req, max_afford)
        stop_mult  = self.tl_cfg.get('stop_loss_atr_mult', 1.5)
        tp_mult    = self.tl_cfg.get('take_profit_atr_mult', 2.0)

        self.log(f"CASH={self.broker.getcash():.2f}, PRICE={self.data.close[0]:.2f}, size={size:.6f}")

        # 9) Bracket orders
        if sig ==  1:
            entry, stop, limit = self.buy_bracket(
                size       = size,
                price      = self.data.close[0],
                stopprice  = self.data.close[0] - stop_mult * self.atr[0],
                limitprice = self.data.close[0] + tp_mult   * self.atr[0],
            )

        elif sig == -1:
            entry, stop, limit = self.sell_bracket(
                size       = size,
                price      = self.data.close[0],
                stopprice  = self.data.close[0] + stop_mult * self.atr[0],
                limitprice = self.data.close[0] - tp_mult   * self.atr[0],
            )
        # 10) Periodic logging
        if dt.minute % 15 == 0:
            self.log(f"Signal={sig}, PosFac={pos_fac:.3f}, Edge={edge:.3f}, Thresh={thr:.3f}")

    def get_metrics(self):
        return pd.DataFrame({
            'close':      self.closes,
            'exp_return': self.exp_returns,
            'edge_norm':  self.edge_norms,
            'threshold':  self.thresholds,
            'signal':     self.signals,
            'position':   self.positions_log,
        })
