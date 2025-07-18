# src/backtesting.strategy.py


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
        self.cfg = self.params.config
        self.tl_cfg = self.cfg['trading_logic']
        self.model_cfg = self.cfg['model']
        self.models = {}
        self.feature_names = []

        # Load models
        for horizon, model_path in self.tl_cfg['model_paths'].items():
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            self.models[horizon] = model

        # Load feature list
        with open(self.model_cfg['features'], 'r') as f:
            self.feature_names = json.load(f)

        # Metrics tracking
        self.exp_returns = []
        self.vol_window = self.tl_cfg['vol_window']
        self.edge_norms = []
        self.thresholds = []
        self.signals = []
        self.positions_log = []
        self.pnls = []
        self.closes = []
        self.ret_buffer = []

        # Signal persistence buffer
        self.signal_buffer = deque(maxlen=self.tl_cfg.get('persistence', 3))

        # ATR-based stops
        atr_period = self.tl_cfg.get('atr_period', 14)
        self.atr = bt.indicators.ATR(self.data, period=atr_period)

        # Log configured ATR multipliers
        stop_m = self.tl_cfg.get('stop_loss_atr_mult', 1.5)
        tp_m   = self.tl_cfg.get('take_profit_atr_mult', 2.0)
        self.log(f"Using stop_mult={stop_m}, tp_mult={tp_m}")

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f"{dt.isoformat()} - {txt}")

    def notify_order(self, order):
        status_name = order.Status[order.status] if hasattr(order, 'Status') else order.status

        if order.status in [order.Completed]:
            action = "BUY" if order.isbuy() else "SELL"
            price  = order.executed.price
            size   = order.executed.size
            self.log(f"{action} EXECUTED: Price={price}, Size={size}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # For these, executed.price is 0 or unavailable, so just show the status
            self.log(f"⚠️ Order {status_name}")


    def next(self):
        dt = self.datas[0].datetime.datetime(0)

        # 1) Build feature row → predict
        row = {f: getattr(self.datas[0], f)[0] for f in self.feature_names}
        df  = pd.DataFrame([row])
        preds = [m.predict(df)[0] for m in self.models.values()]
        exp_return = float(np.mean(preds))

        # 2) Rolling volatility
        if self.closes:
            ret = (self.data.close[0] - self.closes[-1]) / self.closes[-1]
            self.ret_buffer.append(ret)
        self.closes.append(self.data.close[0])

        if len(self.ret_buffer) >= self.tl_cfg['vol_window']:
            vol = pd.Series(self.ret_buffer[-self.tl_cfg['vol_window']:]).std()
        else:
            vol = np.nan

        # 3) Normalize edge & threshold
        if not np.isnan(vol) and vol > 0:
            edge_norm = exp_return / vol
            threshold = (self.tl_cfg['fee_rate'] * self.tl_cfg.get('threshold_mult', 1.0)) / vol
        else:
            edge_norm, threshold = 0.0, float('inf')

        # 4) Raw signal + persistence
        signal = 1 if edge_norm >= threshold else (-1 if edge_norm <= -threshold else 0)
        self.signal_buffer.append(signal)

        # 5) Scale position factor
        conf      = abs(edge_norm)
        max_conf  = max(self.edge_norms) if self.edge_norms else 1e-8
        scaled_position = (conf / max(max_conf, 1e-8)) * self.tl_cfg['max_position'] * signal

        # 6) Append *all* metrics unconditionally
        self.exp_returns.append(exp_return)
        self.edge_norms.append(edge_norm)
        self.thresholds.append(threshold)
        self.signals.append(signal)
        self.positions_log.append(scaled_position)
        self.pnls.append(self.position.size * exp_return * self.data.close[0])

        # 7) Only place orders if we're in hours and long
        in_hours = 8 <= dt.hour <= 20
        if not in_hours or scaled_position <= 0 or not any(s == 1 for s in self.signal_buffer):
            # skip trading logic, but metrics are safely recorded
            return

         # 1) Cancel any leftover bracket (entry + stop + limit)
        if getattr(self, 'bracket_orders', None):
            for o in self.bracket_orders:
                self.cancel(o)
            self.bracket_orders = []

        # 8) Compute actual size and bracket parameters
        size_req = (self.broker.getcash() * abs(scaled_position)) / self.data.close[0]
        max_affordable = self.broker.getcash() / self.data.close[0]
        size = min(size_req, max_affordable)

        stop_mult = self.tl_cfg.get('stop_loss_atr_mult', 1.5)
        tp_mult   = self.tl_cfg.get('take_profit_atr_mult', 2.0)

        self.log(f"{dt.isoformat()} ▶️ CASH={self.broker.getcash():.2f}, "
                f"PRICE={self.data.close[0]:.2f}, size={size:.6f}")

        # 9) Place bracket
        self.buy_bracket(
            size      = size,
            price     = self.data.close[0],
            stopprice = self.data.close[0] - stop_mult * self.atr[0],
            limitprice= self.data.close[0] + tp_mult   * self.atr[0],
        )

        # 10) Optional periodic logging
        if dt.minute % 15 == 0:
            self.log(f"Signal={signal}, PosFactor={scaled_position:.3f}, "
                    f"Edge={edge_norm:.3f}, Thresh={threshold:.3f}")

    def get_metrics(self):
        return pd.DataFrame({
            'close':       self.closes,
            'exp_return':  self.exp_returns,
            'edge_norm':   self.edge_norms,
            'threshold':   self.thresholds,
            'signal':      self.signals,
            'position':    self.positions_log,
            'pnl':         self.pnls
        })
