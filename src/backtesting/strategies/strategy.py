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
        'config' : None, 
        'enable_timed_exit' : True,
        'enable_persistence' : True,
        'enable_quantile' : True,
     }

    def __init__(self):
        # --- CONFIG & MODELS ---
        self.cfg    = self.p.config
        self.tl     = self.cfg['trading_logic']
        self.last_trade_info = {}  # ADDED: per-trade facts captured at fills

        # load models
        self.models         = {}
        self.feature_names  = {}
        for h, path in self.tl['model_paths'].items():
            m = xgb.XGBRegressor()
            m.load_model(path)
            ih = int(h)
            self.models[ih]        = m
            self.feature_names[ih] = m.get_booster().feature_names

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
        self.closes        = []
        self.ret_buffer    = []
        self.exp_returns   = []
        self.edge_norms    = []
        self.thresholds    = []
        self.signals       = []
        self.signal_buffer = deque(maxlen=self.tl.get('persistence', 1))

        self.atr = bt.indicators.ATR(self.data, period=self.tl.get('atr_period',14))
        self.sma = bt.indicators.SMA(self.data.close,
                                     period=self.cfg.get('features',{}).get('sma_window',20))
        self.rsi = bt.indicators.RSI(self.data.close,
                                     period=self.cfg.get('features',{}).get('rsi_window',14))
        self.tp_buffer  = deque(maxlen=self.vol_window)
        self.vol_buffer = deque(maxlen=self.vol_window)
        self.std        = bt.indicators.StdDev(self.data.close, period=self.vol_window)

        print(f"{datetime.utcnow().isoformat()} - KrakenStrategy initialized")

        # ADDED: state for deferred OCO children (prevents same-bar exits)
        self.entry_order    = None
        self.stop_order     = None
        self.limit_order    = None
        self.children_armed = False   # True right after entry fills; children placed next bar
        self.entry_bar      = None    # bar index at entry
        self.entry_fill_bar = None    # bar index when broker reports entry Completed
        self.pending_exit_reason = None  # reason to use when we intentionally flatten

    def notify_order(self, order):
        return notifies.notify_order(self, order)
    
    def notify_trade(self, trade):
        return notifies.notify_trade(self, trade)

    def next(self):
        self._clear_stale_orders()
        if self.p.enable_timed_exit:
            self._check_timed_exit()

        # ADDED: place OCO children on the bar AFTER the entry filled -> prevents same-bar TP/SL hits
        if self.children_armed and self.position and self.entry_fill_bar is not None and len(self) > self.entry_fill_bar:
            #print(f"DEBUG: Position size before placing children: {self.position.size}")
            sz = abs(self.position.size)
            #print(f"DEBUG: Child order size: {sz}")
            if self.position.size > 0:
                self.stop_order  = self.sell(size=sz, exectype=bt.Order.Stop,  price=self.armed_stop)
                self.limit_order = self.sell(size=sz, exectype=bt.Order.Limit, price=self.armed_limit, oco=self.stop_order)
            else:
                self.stop_order  = self.buy(size=sz,  exectype=bt.Order.Stop,  price=self.armed_stop)
                self.limit_order = self.buy(size=sz,  exectype=bt.Order.Limit, price=self.armed_limit, oco=self.stop_order)
            self.orders = [o for o in (self.stop_order, self.limit_order) if o is not None]
            self.children_armed = False
            #print(f"{self.data.datetime.datetime(0).isoformat()} – 🛡️ Children placed: stop={self.armed_stop:.6f}, tp={self.armed_limit:.6f}")

        if self._in_flight():
            return
        
        vol = np.std(self.ret_buffer[-self.vol_window:]) if len(self.ret_buffer) >= self.vol_window else 0.0
        self._update_buffers()

        exp_r = self._predict_return()
        self.exp_returns.append(exp_r)            
        edge, thr, sig = self._compute_signal(exp_r, vol)

        self._log_bar_metrics(exp_r, edge, vol, sig, thr)

        # if len(self.exp_returns) % 50 == 0:  # log every ~50 bars
        #     print(f"{self.data.datetime.datetime(0)} | exp_r={exp_r:.8f} | edge={edge:.8f} | thr={thr:.8f} | sig={sig}")


        self.edge_norms.append(edge)               # <-- keep these lists fresh
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

    # helper methods
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
        
    def _predict_return(self):
        preds = []
        for h, m in self.models.items():
            row = self._make_feature_row(h)
            preds.append(m.predict(pd.DataFrame([row]))[0])
        weights = np.array(self.cfg['trading_logic']['horizon_weights'], dtype=float)[:len(preds)]
        return np.dot(preds, weights) / weights.sum() if weights.sum() else float(np.mean(preds))

    def _compute_signal(self, exp_r, vol):
        eps = 1e-8
        vol = max(vol, eps)
        edge = exp_r / vol
        thr  = (self.fee_rate * self.threshold_mult)
        sig  = 1 if edge >= thr else -1 if edge <= -thr else 0

        # --- DEBUG ---
        if np.random.rand() < 0.001:  # sample 0.1% of bars to avoid spam
            print(f"[DEBUG] exp_r={exp_r:.6f}, vol={vol:.6f}, edge={edge:.3f}, thr={thr:.3f}, sig={sig}")


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
        # sizing logic unchanged
        recent = self.edge_norms[-50:]
        ref = np.percentile(np.abs(recent), 90) if recent else 1e-4
        size_fac = min(abs(edge) / max(ref, 1e-4), 1.0) * self.max_position * sig
        cash = self.broker.getcash()
        raw = (cash * size_fac) / self.data.close[0]
        size = min(round(raw, 8), cash / self.data.close[0])

        # --- DEBUG ---
        if np.random.rand() < 0.001:
            print(f"[DEBUG] edge={edge:.3f}, ref={ref:.3f}, size_fac={size_fac:.4f}, raw={raw:.6f}, size={size}")



        return size if sig * size > 0 and abs(size) >= self.min_trade_size else None

    def _place_bracket(self, size, price):
        """
        CHANGED: ENTRY-ONLY. We deliberately do NOT submit children here.
        Children are armed in notify_order at entry fill and submitted next bar in next().
        This removes same-bar bracket exits.
        """
        if size > 0:
            self.entry_order = self.buy(size=abs(size))   # market entry
        else:
            self.entry_order = self.sell(size=abs(size))
        self.orders = [self.entry_order]

    def get_trade_log_df(self):
        return pd.DataFrame([t.__dict__ for t in self.trade_log])

    def _get_feature_value(self, feat_name: str) -> float:
        """
        Read feature from EngineeredData feed.
        - lower-case
        - replace '.' with '_' to match feed line names
        """
        lf = feat_name.lower()
        attr = lf.replace('.', '_')
        line = getattr(self.data, attr, None) or getattr(self.data, lf, None)
        if line is None:
            raise KeyError(f"Missing feature '{feat_name}' (tried '{attr}'). Check EngineeredData lines/params.")
        return float(line[0])

    def _make_feature_row(self, h: int):
        feats = self.feature_names[h]  # use horizon-specific features from the booster
        return {f: self._get_feature_value(f) for f in feats}


    def _log_bar_metrics(self, exp_r, edge, vol, sig, thr):
        """Append per-bar diagnostics for model accuracy analysis."""
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
        import pandas as pd
        df = pd.DataFrame(self.metrics_buffer)
        # Rename to match old schema
        df = df.rename(columns={'exp_r': 'exp_return', 'edge': 'edge_norm'})
        # Ensure required columns exist
        if 'position' not in df.columns:
            df['position'] = 0.0

        # print("DEBUG metrics_buffer length:", len(self.metrics_buffer))
        # if self.metrics_buffer:
        #     print("DEBUG first metrics row:", self.metrics_buffer[0])

        return df

