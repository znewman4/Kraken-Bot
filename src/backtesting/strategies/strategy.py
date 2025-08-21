# src/backtesting/strategy.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # repo root

import backtrader as bt
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from src.backtesting.engine import notifies
from src.backtesting.engine.feature_bridge import FeatureBridge   # ← NEW


class KrakenStrategy(bt.Strategy):

    @dataclass
    class TradeLogEntry:
        entry_time: str
        exit_time: str
        entry_price: float
        exit_price: float
        position_size: float
        predicted_exp_r: float
        edge: float
        volatility: float
        net_pnl: float
        pnl_per_unit: float
        stop_hit: bool
        take_profit_hit: bool
        hold_time_mins: float
        atr: float
        stop_dist: float
        tp_dist: float
        bar_high: float
        bar_low: float
        bar_close: float
        signal: int
        entry_bar: int
        exit_bar: int
        bars_held: int

    params = {
        'config': None,
        'enable_timed_exit': False,
        'enable_persistence': False,
        'enable_quantile': False,
    }

    def __init__(self):
        # --- CONFIG ---
        self.cfg = self.p.config
        self.tl  = self.cfg['trading_logic']
        self.last_trade_info = {}

        # --- MODELS ---
        self.models = {}
        for h, path in self.tl['model_paths'].items():
            ih = int(h)
            m = xgb.XGBRegressor()
            m.load_model(path)
            self.models[ih] = m

        # parameters
        self.max_hold_bars   = int(self.tl.get('max_hold_bars', 29))
        self.quantile_window = int(self.tl.get('quantile_window', 1000))
        self.entry_quantile  = float(self.tl.get('entry_quantile', 0.8))
        self.stop_mult       = float(self.tl.get('stop_loss_atr_mult', 4.0))
        self.tp_mult         = float(self.tl.get('take_profit_atr_mult', 6.0))
        self.fee_rate        = float(self.tl.get('fee_rate', 0.002))
        self.threshold_mult  = float(self.tl.get('threshold_mult', 0.05))
        self.vol_window      = int(self.tl.get('vol_window', 20))
        self.min_trade_size  = float(self.tl.get('min_trade_size', 0.01))
        self.max_position    = float(self.tl.get('max_position', 1.0))
        self.tick_size       = float(self.tl.get('tick_size', 0.01))

        # state & logs
        self.trade_log      = []
        self.orders         = []
        self.pnls           = []

        # buffers and indicators
        self.metrics_buffer = []
        self.closes         = []
        self.ret_buffer     = []
        self.exp_returns    = []
        self.edge_norms     = []
        self.thresholds     = []
        self.signals        = []
        self.signal_buffer  = deque(maxlen=self.tl.get('persistence', 1))

        self.atr = bt.indicators.ATR(self.data, period=self.tl.get('atr_period', 14))
        self.sma = bt.indicators.SMA(self.data.close,
                                     period=self.cfg.get('features', {}).get('sma_window', 20))
        self.rsi = bt.indicators.RSI(self.data.close,
                                     period=self.cfg.get('features', {}).get('rsi_window', 14))
        self.tp_buffer  = deque(maxlen=self.vol_window)
        self.vol_buffer = deque(maxlen=self.vol_window)
        self.std        = bt.indicators.StdDev(self.data.close, period=self.vol_window)

        print(f"{datetime.utcnow().isoformat()} - KrakenStrategy initialized")

        # state for deferred OCO children
        self.entry_order    = None
        self.stop_order     = None
        self.limit_order    = None
        self.children_armed = False
        self.entry_bar      = None
        self.entry_fill_bar = None
        self.pending_exit_reason = None

        # 🔗 Build the feature bridge (does parity checks & mapping)
        self.bridge = FeatureBridge(
            bt_data=self.data,
            models_by_horizon=self.models,
            model_paths=self.tl['model_paths'],
            verbose=True
        )

    # -------- Backtrader plumbing --------
    def notify_order(self, order):
        return notifies.notify_order(self, order)

    def notify_trade(self, trade):
        return notifies.notify_trade(self, trade)

    def next(self):
        self._clear_stale_orders()
        if self.p.enable_timed_exit:
            self._check_timed_exit()

        if self.children_armed and self.position and self.entry_fill_bar is not None and len(self) > self.entry_fill_bar:
            sz = abs(self.position.size)
            if self.position.size > 0:
                self.stop_order  = self.sell(size=sz, exectype=bt.Order.Stop,  price=self.armed_stop)
                self.limit_order = self.sell(size=sz, exectype=bt.Order.Limit, price=self.armed_limit, oco=self.stop_order)
            else:
                self.stop_order  = self.buy(size=sz,  exectype=bt.Order.Stop,  price=self.armed_stop)
                self.limit_order = self.buy(size=sz,  exectype=bt.Order.Limit, price=self.armed_limit, oco=self.stop_order)
            self.orders = [o for o in (self.stop_order, self.limit_order) if o is not None]
            self.children_armed = False

        if self._in_flight():
            return

        vol = np.std(self.ret_buffer[-self.vol_window:]) if len(self.ret_buffer) >= self.vol_window else 0.0
        self._update_buffers()

        # ⭐️ Call into the bridge for aligned prediction
        exp_r = self._predict_return()
        self.exp_returns.append(exp_r)
        edge, thr, sig = self._compute_signal(exp_r, vol)

        self._log_bar_metrics(exp_r, edge, vol, sig, thr)

        self.edge_norms.append(edge)
        self.thresholds.append(thr)
        self.signals.append(sig)
        self.signal_buffer.append(sig)

        if sig == 0:
            return
        if self.p.enable_persistence and not self._pass_persistence(sig):
            return
        if self.p.enable_quantile and not self._pass_quantile(exp_r, sig):
            return

        size = self._size_position(edge, sig)
        if size is None:
            return

        price = float(self.data.close[0])
        self._place_bracket(size, price)

    # -------- Helpers --------
    def _clear_stale_orders(self):
        if self.orders and not self.position:
            self.orders = []

    def _check_timed_exit(self):
        if self.position and hasattr(self, 'last_trade_info') and 'entry_bar' in self.last_trade_info:
            bars_held = len(self) - self.last_trade_info['entry_bar']
            if bars_held >= self.max_hold_bars:
                for o in list(self.orders):
                    self.cancel(o)
                self.orders = []
                self.pending_exit_reason = "Timed"
                self.close()

    def _in_flight(self):
        return bool(self.position or self.orders)

    def _update_buffers(self):
        prev_close = self.closes[-1] if self.closes else self.data.close[0]
        self.ret_buffer.append((self.data.close[0] - prev_close) / prev_close if prev_close else 0.0)
        self.closes.append(self.data.close[0])

    # ==== delegate to FeatureBridge ====
    def _predict_return(self) -> float:
        weights = self.cfg['trading_logic']['horizon_weights']
        return self.bridge.predict_return(weights)

    # ==== signals/sizing/logging (unchanged) ====
    def _compute_signal(self, exp_r, vol):
        eps = 1e-8
        vol = max(vol, eps)
        edge = exp_r / vol
        thr  = (self.fee_rate * self.threshold_mult) / vol if vol > 0 else float('inf')
        sig  = 1 if edge >= thr else -1 if edge <= -thr else 0
        return edge, thr, sig

    def _pass_persistence(self, sig):
        return len(self.signal_buffer) == self.signal_buffer.maxlen and all(s == sig for s in self.signal_buffer)

    def _pass_quantile(self, exp_r, sig):
        if len(self.exp_returns) < self.quantile_window:
            return True
        hist = np.array(self.exp_returns[-self.quantile_window:])
        long_thr = np.quantile(hist, self.entry_quantile)
        short_thr = np.quantile(hist, 1 - self.entry_quantile)
        return not ((sig == 1 and exp_r < long_thr) or (sig == -1 and exp_r > short_thr))

    def _size_position(self, edge, sig):
        recent = self.edge_norms[-50:]
        ref = np.percentile(np.abs(recent), 90) if recent else 1e-4
        size_fac = min(abs(edge) / max(ref, 1e-4), 1.0) * self.max_position * sig
        cash = self.broker.getcash()
        raw = (cash * size_fac) / self.data.close[0]
        size = min(round(raw, 8), cash / self.data.close[0])
        return size if sig * size > 0 and abs(size) >= self.min_trade_size else None

    def _place_bracket(self, size, price):
        if size > 0:
            self.entry_order = self.buy(size=abs(size))
        else:
            self.entry_order = self.sell(size=abs(size))
        self.orders = [self.entry_order]

    def get_trade_log_df(self):
        return pd.DataFrame([t.__dict__ for t in self.trade_log])

    def _log_bar_metrics(self, exp_r, edge, vol, sig, thr):
        self.metrics_buffer.append({
            'datetime': self.data.datetime.datetime(0),
            'close': float(self.data.close[0]),
            'exp_r': exp_r,
            'edge': edge,
            'volatility': vol,
            'signal': sig,
            'threshold': thr
        })

    def get_metrics(self):
        df = pd.DataFrame(self.metrics_buffer)
        df = df.rename(columns={'exp_r': 'exp_return', 'edge': 'edge_norm'})
        if 'position' not in df.columns:
            df['position'] = 0.0
        return df
