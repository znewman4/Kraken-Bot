print("Loading Strategy 2...")

import backtrader as bt
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from collections import deque


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

        # Load feature list
        with open(self.model_cfg['features'], 'r') as f:
            self.feature_names = json.load(f)

        # Data trackers & persistence
        self.closes        = []
        self.ret_buffer    = []
        self.exp_returns   = []
        self.edge_norms    = []           # Changed: moved metrics init
        self.thresholds    = []
        self.signals       = []
        self.positions_log = []
        self.pnls          = []

        # Signal persistence buffer
        self.signal_buffer = deque(maxlen=self.tl_cfg.get('persistence', 3))  # Changed: increased default persistence

        # ATR for stops / take-profit
        atr_p = self.tl_cfg.get('atr_period', 14)
        self.atr = bt.indicators.ATR(self.data, period=atr_p)

        # Hold last bracket orders
        self.bracket_orders = []           # Changed: store bracket orders

        sm, tm = (self.tl_cfg['stop_loss_atr_mult'],
                  self.tl_cfg['take_profit_atr_mult'])
        self.log(f"Using stop_mult={sm}, tp_mult={tm}")

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f"{dt.isoformat()} - {txt}")

    def notify_order(self, order):
        status = order.Status[order.status] if hasattr(order, 'Status') else order.status
        if order.status == bt.Order.Completed:
            act = 'BUY' if order.isbuy() else 'SELL'
            self.log(f"{act} EXECUTED: Price={order.executed.price}, Size={order.executed.size}")
        elif order.status in (bt.Order.Canceled, bt.Order.Margin, bt.Order.Rejected):
            self.log(f"⚠️ Order {status}")

    def next(self):
        dt = self.datas[0].datetime.datetime(0)

        # 1) Predict
        row = {f: getattr(self.datas[0], f)[0] for f in self.feature_names}
        df  = pd.DataFrame([row])
        preds = [m.predict(df)[0] for m in self.models.values()]
        exp_r = float(np.mean(preds))

        # 2) Rolling vol
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
            edge  = exp_r / vol
            thr   = (self.tl_cfg['fee_rate'] * self.tl_cfg.get('threshold_mult', 1.0)) / vol
        else:
            edge, thr = 0.0, float('inf')

        # 4) Signal + persistence
        sig = 1 if edge >= thr else (-1 if edge <= -thr else 0)
        self.signal_buffer.append(sig)

        # 5) Scale position factor
        conf     = abs(edge)
        max_conf = max(self.edge_norms) if self.edge_norms else 1e-8
        pos_fac  = (conf / max(max_conf, 1e-8)) * self.tl_cfg['max_position'] * sig

        # 6) _Always_ record metrics **before** any early return                # Changed: unconditional metrics logging
        self.exp_returns.append(exp_r)
        self.edge_norms.append(edge)
        self.thresholds.append(thr)
        self.signals.append(sig)
        self.positions_log.append(pos_fac)
        self.pnls.append(self.position.size * exp_r * self.data.close[0])

        # 7) Skip if outside trading hours or not a long signal or no persistence # Changed: stricter entry gating
        if not (8 <= dt.hour <= 20) or pos_fac <= 0 or not any(s == 1 for s in self.signal_buffer):
            return

        # 8) Compute actual size and clamp to available cash                         # Changed: size clamp
        size_req     = (self.broker.getcash() * abs(pos_fac)) / self.data.close[0]
        max_afford   = self.broker.getcash() / self.data.close[0]
        size         = min(size_req, max_afford)
        stop_mult    = self.tl_cfg.get('stop_loss_atr_mult', 1.5)
        tp_mult      = self.tl_cfg.get('take_profit_atr_mult', 2.0)

        self.log(f"{dt.isoformat()} ▶️ CASH={self.broker.getcash():.2f}, PRICE={self.data.close[0]:.2f}, size={size:.6f}")   # Changed: enhanced log

        # 9) Place bracket **and** capture orders                                 # Changed: unpack bracket
        entry, stop, limit = self.buy_bracket(
            size      = size,
            price     = self.data.close[0],
            stopprice = self.data.close[0] - stop_mult * self.atr[0],
            limitprice= self.data.close[0] + tp_mult   * self.atr[0],
        )
        self.bracket_orders = [entry, stop, limit]  # Changed: store bracket orders

        # 10) Periodic logging unchanged
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
            'pnl':        self.pnls
        })
