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
        # Time-of-day filter
        dt = self.datas[0].datetime.datetime(0)
        if not (8 <= dt.hour <= 20):
            return

        # Ensure all features are present
        missing = [f for f in self.feature_names if not hasattr(self.datas[0], f)]
        if missing:
            self.log(f"⚠️ Skipping row. Missing features: {missing}")
            return

        # Build feature row and predict
        row = {name: getattr(self.datas[0], name)[0] for name in self.feature_names}
        df = pd.DataFrame([row])
        preds = [model.predict(df)[0] for model in self.models.values()]
        exp_return = np.mean(preds)
        self.exp_returns.append(exp_return)

        # Rolling volatility
        if self.closes:
            ret = (self.data.close[0] - self.closes[-1]) / self.closes[-1]
            self.ret_buffer.append(ret)
        self.closes.append(self.data.close[0])
        volatility = pd.Series(self.ret_buffer[-self.vol_window:]).std() if len(self.ret_buffer) >= self.vol_window else np.nan

        # Normalize edge & threshold
        if volatility and volatility > 0:
            edge_norm = exp_return / volatility
            threshold = (self.tl_cfg['fee_rate'] * self.tl_cfg.get('threshold_mult', 1.0)) / volatility
        else:
            edge_norm = 0
            threshold = np.inf

        # Determine raw signal
        signal = 1 if edge_norm >= threshold else (-1 if edge_norm <= -threshold else 0)
        self.signal_buffer.append(signal)

        # Compute scaled position factor
        confidence = abs(edge_norm)
        max_conf = max(self.edge_norms) if self.edge_norms else 1e-8
        scaled_conf = confidence / max(max_conf, 1e-8)
        scaled_position = scaled_conf * self.tl_cfg['max_position'] * signal

        # Entry with ATR bracket orders
        if all(s == 1 for s in self.signal_buffer) and not self.position:
            size = (self.broker.getcash() * abs(scaled_position)) / self.data.close[0]
            stop_mult = self.tl_cfg.get('stop_loss_atr_mult', 1.5)
            tp_mult   = self.tl_cfg.get('take_profit_atr_mult', 2.0)
            self.buy_bracket(
                size=size,
                price=self.data.close[0],
                stopprice=self.data.close[0] - stop_mult * self.atr[0],
                limitprice=self.data.close[0] + tp_mult * self.atr[0]
            )

        # Track metrics and log
        self.edge_norms.append(edge_norm)
        self.thresholds.append(threshold)
        self.signals.append(signal)
        self.positions_log.append(scaled_position)
        pnl = self.position.size * exp_return * self.data.close[0]
        self.pnls.append(pnl)

        if dt.minute % 15 == 0:
            self.log(f"Signal={signal}, PosFactor={scaled_position:.3f}, Edge={edge_norm:.3f}, Thresh={threshold:.3f}")

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
