import backtrader as bt
import pandas as pd
import numpy as np
import xgboost as xgb
import json


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

        # Internal state tracking
        self.exp_returns = []
        self.vol_window = self.tl_cfg['vol_window']
        self.edge_norms = []
        self.thresholds = []
        self.signals = []
        self.positions_log = []
        self.pnls = []
        self.closes = []
        self.ret_buffer = []

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        #print(f"{dt.isoformat()} - {txt}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            action = "BUY" if order.isbuy() else "SELL"
            self.log(f"{action} EXECUTED: Price={order.executed.price}, Size={order.executed.size}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"⚠️ Order {order.Status[order.status]} at Price={order.created.price}")

    def next(self):
        # Fetch feature row
        missing_features = [name for name in self.feature_names if not hasattr(self.datas[0], name)]
        if missing_features:
            self.log(f"⚠️ Skipping row. Missing features: {missing_features}")
            return

        row = {name: getattr(self.datas[0], name)[0] for name in self.feature_names}

    
        df = pd.DataFrame([row])

        # Predict using all horizon models
        preds = [model.predict(df)[0] for model in self.models.values()]
        exp_return = np.mean(preds)
        self.exp_returns.append(exp_return)

        # Rolling volatility using close returns
        if len(self.closes) >= 1:
            ret = (self.data.close[0] - self.closes[-1]) / self.closes[-1]
            self.ret_buffer.append(ret)

        self.closes.append(self.data.close[0])

        if len(self.ret_buffer) >= self.vol_window:
            volatility = pd.Series(self.ret_buffer[-self.vol_window:]).std()
        else:
            volatility = np.nan

        # Normalize edge
        if volatility and volatility > 0:
            edge_norm = exp_return / volatility
            threshold = (self.tl_cfg['fee_rate'] * 0.5) / volatility
        else:
            edge_norm = 0
            threshold = np.inf

        signal = 0

        buffer = 1.5
        if edge_norm >= threshold*buffer:
            signal = 1
        elif edge_norm <= -threshold*buffer:
            signal = -1

        # Position sizing
        confidence = abs(edge_norm)
        max_conf = max(self.edge_norms) if self.edge_norms else 1.0
        max_conf = max(confidence, max_conf, 1e-8)  # avoid div by zero
        scaled_position = (confidence / max_conf) * self.tl_cfg['max_position'] * signal

        # Trade logic
        stake = self.tl_cfg.get('btc_stake', 1.0)
        predicted_dollar_move = exp_return * self.data.close[0]
        pnl = self.position.size * predicted_dollar_move * stake

        if signal == 1 and not self.position:
            size = (self.broker.getcash() * abs(scaled_position)) / self.data.close[0]
            self.buy(size=size)
        elif signal == -1 and self.position:
            self.close()

        # Log signals and diagnostics
        dt = self.datas[0].datetime.datetime(0)
        if dt.minute % 15 == 0:
            self.log(f"Signal={signal}, Position={self.position.size}, Scaled={scaled_position:.4f}, Conf={confidence:.4f}, Edge={edge_norm:.4f}, Thresh={threshold:.4f}")

        # Track for metrics
        self.edge_norms.append(edge_norm)
        self.thresholds.append(threshold)
        self.signals.append(signal)
        self.positions_log.append(scaled_position)
        self.pnls.append(pnl)

    def get_metrics(self):
        return pd.DataFrame({
            'close': self.closes,
            'exp_return': self.exp_returns,
            'edge_norm': self.edge_norms,
            'threshold': self.thresholds,
            'signal': self.signals,
            'position': self.positions_log,
            'pnl': self.pnls
        })
