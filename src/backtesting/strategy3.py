# src/backtesting/strategy3.py
import backtrader as bt
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import deque
from datetime import datetime

class KrakenStrategy(bt.Strategy):
    params = (
        ('config', None),
    )

    def __init__(self):
        # --- CONFIG & MODELS ---
        self.cfg        = self.p.config
        self.tl_cfg     = self.cfg['trading_logic']
        self.models     = {}
        self.feature_names_map = {}

        # Load each XGB model and remember its feature names
        for horizon, path in self.tl_cfg['model_paths'].items():
            m = xgb.XGBRegressor()
            m.load_model(path)
            self.models[horizon] = m
            self.feature_names_map[horizon] = m.get_booster().feature_names

        # --- DATA TRACKERS ---
        self.closes        = []
        self.ret_buffer    = []
        self.exp_returns   = []
        self.edge_norms    = []
        self.thresholds    = []
        self.signals       = []
        self.positions_log = []
        self.pnls          = []

        # signal persistence buffer
        persistence = self.tl_cfg.get('persistence', 3)
        self.signal_buffer = deque(maxlen=persistence)

        # --- INDICATORS ---
        # ATR for stops & take‐profit
        atr_period = self.tl_cfg.get('atr_period', 14)
        self.atr = bt.indicators.ATR(self.data, period=atr_period)

        # SMA (for feature lookup)
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

        # VWAP buffers
        self.vwap_period = self.tl_cfg.get('vol_window', 20)
        self.tp_buffer   = deque(maxlen=self.vwap_period)
        self.vol_buffer  = deque(maxlen=self.vwap_period)

        print(f"{datetime.utcnow().isoformat()} - KrakenStrategy initialized")

    def notify_trade(self, trade):
        if trade.isclosed:
            pnl = trade.pnlcomm
            print(f"{self.datas[0].datetime.datetime(0).isoformat()} - TRADE CLOSED | Net PnL={pnl:.2f}")
            self.pnls.append(pnl)

    def notify_order(self, order):
        # --- Completed executions ---
        if order.status == bt.Order.Completed:
            side = 'BUY' if order.isbuy() else 'SELL'
            ts   = self.datas[0].datetime.datetime(0).isoformat()
            print(f"{ts} – {side} EXECUTED @ {order.executed.price:.5f} qty {order.executed.size:.6f}")

        # --- Any cancellations, rejections, margin failures ---
        elif order.status in (bt.Order.Canceled, bt.Order.Margin, bt.Order.Rejected):
            ts = self.datas[0].datetime.datetime(0).isoformat()
            status_name = order.getstatusname()
            print(f"{ts} – ⚠️ Order {status_name}")

    def next(self):
        # 1) VWAP calc
        tp = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3.0
        self.tp_buffer.append(tp)
        self.vol_buffer.append(self.data.volume[0])
        vwap = sum(p * v for p, v in zip(self.tp_buffer, self.vol_buffer)) / max(sum(self.vol_buffer), 1)

        # 2) Make preds
        preds   = []
        weights = self.tl_cfg.get('horizon_weights', [])
        horizons = list(self.tl_cfg['model_paths'].keys())

        for idx, horizon in enumerate(horizons):
            m     = self.models[horizon]
            feats = self.feature_names_map[horizon]
            row   = {}
            for f in feats:
                lf = f.lower()
                if lf == 'sma':
                    row[f] = float(self.sma[0])
                elif lf == 'rsi':
                    row[f] = float(self.rsi[0])
                elif lf == 'vwap':
                    row[f] = float(vwap)
                else:
                    row[f] = float(getattr(self.data, f)[0])
            dfm = pd.DataFrame([row])
            preds.append(m.predict(dfm)[0])

        # Weighted average of preds
        w = np.array(weights[:len(preds)], dtype=float)
        if w.sum() == 0:
            exp_r = float(np.mean(preds))
        else:
            exp_r = float(np.dot(preds, w) / w.sum())


        # 3) Rolling vol & edge
        if self.closes:
            r = (self.data.close[0] - self.closes[-1]) / self.closes[-1]
            self.ret_buffer.append(r)
        self.closes.append(self.data.close[0])

        if len(self.ret_buffer) >= self.tl_cfg['vol_window']:
            vol = pd.Series(self.ret_buffer[-self.tl_cfg['vol_window']:]).std()
        else:
            vol = np.nan

        if vol and not np.isnan(vol):
            edge = exp_r / vol
            thr  = (self.tl_cfg['fee_rate'] * self.tl_cfg.get('threshold_mult', 1.0)) / vol
        else:
            edge, thr = 0.0, float('inf')

        sig = 1 if edge >= thr else (-1 if edge <= -thr else 0)
        self.signal_buffer.append(sig)

        # record metrics
        self.exp_returns.append(exp_r)
        self.edge_norms.append(edge)
        self.thresholds.append(thr)
        self.signals.append(sig)

        # 4) gating
        dt = self.datas[0].datetime.datetime(0)
        if not (8 <= dt.hour <= 20) or sig == 0:
            return
        if len(self.signal_buffer) < self.signal_buffer.maxlen or not all(s == sig for s in self.signal_buffer):
            return

        # 5) sizing
        pos_fac = (abs(edge) / max(max(self.edge_norms, default=1e-8),1e-8)) * self.tl_cfg['max_position'] * sig
        pos_fac = max(min(pos_fac, self.tl_cfg['max_position']), -self.tl_cfg['max_position'])
        size_req = (self.broker.getcash() * abs(pos_fac)) / self.data.close[0]
        size = min(size_req, self.broker.getcash() / self.data.close[0])

        # clamp to minimal tradable size (8 dp) and skip if too small
        size = round(size, 8)
        if size <= 0:
            return

        # 6) orders
        stop_mult = self.tl_cfg.get('stop_loss_atr_mult', 1.5)
        tp_mult   = self.tl_cfg.get('take_profit_atr_mult', 2.0)

        if sig == 1:
            self.buy_bracket(
                size       = size,
                price      = self.data.close[0],
                stopprice  = self.data.close[0] - stop_mult * self.atr[0],
                limitprice = self.data.close[0] + tp_mult   * self.atr[0],
            )
        elif sig == -1:
            self.sell_bracket(
                size       = size,
                price      = self.data.close[0],
                stopprice  = self.data.close[0] + stop_mult * self.atr[0],
                limitprice = self.data.close[0] - tp_mult   * self.atr[0],
            )

    def get_metrics(self):
        import pandas as pd

        n = len(self.closes)
        # pad positions_log with 0.0 so it's the same length as closes
        positions = self.positions_log + [0.0] * (n - len(self.positions_log))

        return pd.DataFrame({
            'close':      self.closes,
            'exp_return': self.exp_returns,
            'edge_norm':  self.edge_norms,
            'threshold':  self.thresholds,
            'signal':     self.signals,
            'position':   positions,
        })