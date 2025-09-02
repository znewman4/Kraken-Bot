# src/backtesting/strategies/strat_precomputed.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # repo root

import backtrader as bt
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from dataclasses import dataclass

from src.backtesting.engine import notifies


class KrakenStrategy(bt.Strategy):
    """
    Precomputed variant of  live strategy:
    - Reads precomputed 'exp_return' (bps) and optional 'z_edge' directly from the feed.
    - Applies the same gates, sizing, and trade handling as live.
    - Skips model prediction & calibration (already baked into the input).
    """

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
        ev_bps: float = float('nan')
        sigma_bps: float = float('nan')

    params = {
        "config": None,
        "cost_bps": 0.0,   # round-trip costs (bps)
        "enable_timed_exit": True,
        "enable_persistence": True,
        "enable_quantile": True,
    }

    def __init__(self):
        # --- CONFIG ---
        self.cfg = self.p.config
        self.tl  = self.cfg["trading_logic"]
        self.last_trade_info = {}

        # Precomputed feed columns
        self.pre_col   = self.cfg.get("precomputed_col", "exp_return")  # bps
        self.pre_z_col = self.cfg.get("precomputed_z_col", "z_edge")    # unitless (optional)

        # --- Parameters (mirror live) ---
        self.max_hold_bars   = int(self.tl.get("max_hold_bars", 29))
        self.quantile_window = int(self.tl.get("quantile_window", 1000))
        self.entry_quantile  = float(self.tl.get("entry_quantile", 0.8))
        self.stop_mult       = float(self.tl.get("stop_loss_atr_mult", 4.0))
        self.tp_mult         = float(self.tl.get("take_profit_atr_mult", 6.0))
        self.fee_rate        = float(self.tl.get("fee_rate", 0.002))
        self.threshold_mult  = float(self.tl.get("threshold_mult", 0.05))
        self.vol_window      = int(self.tl.get("vol_window", 20))
        self.min_trade_size  = float(self.tl.get("min_trade_size", 0.01))
        self.max_position    = float(self.tl.get("max_position", 1.0))
        self.tick_size       = float(self.tl.get("tick_size", 0.01))
        self.z_threshold     = float(self.tl.get("z_threshold", 0.35))
        self.min_edge_bps    = float(self.tl.get("min_edge_bps", 0.0))
        self.sigma_floor_bps = float(self.tl.get("sigma_floor_bps", 5.0))
        self.min_warmup_bars = int(self.tl.get("min_warmup_bars", self.vol_window))
        self.cooldown_bars   = int(self.tl.get("cooldown_bars", 0))
        self.vote_buffer     = deque(maxlen=self.tl.get("persistence", 1))
        self.last_exit_bar   = -10**9

        # --- State & logs ---
        self.trade_log, self.orders, self.pnls = [], [], []
        self.metrics_buffer, self.closes, self.ret_buffer = [], [], []
        self.exp_returns, self.edge_norms, self.thresholds, self.signals = [], [], [], []

        # --- Indicators (same as live) ---
        self.atr = bt.indicators.ATR(self.data, period=self.tl.get("atr_period", 14))
        self.sma = bt.indicators.SMA(self.data.close,
                                     period=self.cfg.get("features", {}).get("sma_window", 20))
        self.rsi = bt.indicators.RSI(self.data.close,
                                     period=self.cfg.get("features", {}).get("rsi_window", 14))
        self.std = bt.indicators.StdDev(self.data.close, period=self.vol_window)

        print(f"{datetime.utcnow().isoformat()} - KrakenStrategy (precomputed) initialized")

        # Deferred OCO placement
        self.entry_order = self.stop_order = self.limit_order = None
        self.children_armed = False
        self.entry_bar = self.entry_fill_bar = None
        self.pending_exit_reason = None

    # --- Backtrader notifications ---
    def notify_order(self, order): return notifies.notify_order(self, order)
    def notify_trade(self, trade): return notifies.notify_trade(self, trade)

    # --- Main loop ---
    def next(self):
        self._clear_stale_orders()
        if self.p.enable_timed_exit:
            self._check_timed_exit()

        # Place OCO children after entry fill
        if self.children_armed and self.position and self.entry_fill_bar is not None and len(self) > self.entry_fill_bar:
            sz = abs(self.position.size)
            if self.position.size > 0:
                self.stop_order  = self.sell(size=sz, exectype=bt.Order.Stop,  price=self.armed_stop)
                self.limit_order = self.sell(size=sz, exectype=bt.Order.Limit, price=self.armed_limit, oco=self.stop_order)
            else:
                self.stop_order  = self.buy(size=sz, exectype=bt.Order.Stop,  price=self.armed_stop)
                self.limit_order = self.buy(size=sz, exectype=bt.Order.Limit, price=self.armed_limit, oco=self.stop_order)
            self.orders = [o for o in (self.stop_order, self.limit_order) if o is not None]
            self.children_armed = False

        if self._in_flight():
            return

        pre_ok, pre_why = self._pre_entry_gates()

        # Compute sigma BEFORE buffer update
        sigma = np.std(self.ret_buffer[-self.vol_window:]) if len(self.ret_buffer) >= self.vol_window else 0.0
        sigma_bps = max(1e-8, 1e4 * sigma)
        self._update_buffers()

        if not pre_ok:
            self._dbg(f"skip entry: {pre_why}")
            return

        # --- DIFFERENCE: read precomputed exp_bps + z_edge instead of predicting ---
        exp_bps = float(getattr(self.data, self.pre_col)[0])
        if hasattr(self.data, self.pre_z_col):
            z_edge = float(getattr(self.data, self.pre_z_col)[0])
        else:
            z_edge = exp_bps / sigma_bps  # fallback if z not provided

        self._dbg(f"exp_bps={exp_bps:.2f}, z_edge={z_edge:.3f}")
        self.exp_returns.append(exp_bps)

        # Directional vote & EV gate
        thr = self.z_threshold
        vote = 1 if z_edge >= thr else (-1 if z_edge <= -thr else 0)
        self.vote_buffer.append(vote)

        ev_bps = vote * exp_bps - self.p.cost_bps
        sig = vote if (vote != 0 and ev_bps > self.min_edge_bps) else 0

        self._dbg(f"gates → vote={vote}, ev_bps={ev_bps:.2f}, cost={self.p.cost_bps:.1f}, "
                  f"min_edge={self.min_edge_bps}, vote_buf={list(self.vote_buffer)}")

        # Per-bar metrics
        self._log_bar_metrics(exp_bps, z_edge, sigma, sig, thr)

        # Post gates
        post_ok, post_why = self._post_entry_gates(sig, exp_bps, vote)
        if not post_ok:
            self._dbg(f"skip entry: {post_why}")
            return

        # Diagnostics tracking
        self.edge_norms.append(z_edge)
        self.thresholds.append(thr)
        self.signals.append(sig)

        # Sizing
        size = self._size_position(z_edge, sig)
        if size is None:
            return

        price = float(self.data.close[0])
        self._place_bracket(size, price)

        self._entry_snapshot = {
            "exp_bps": float(exp_bps),
            "z_edge":  float(z_edge),
            "sigma_bps": float(sigma_bps),
            "ev_bps": float(ev_bps),
            "sig": int(sig),
        }

    # --- Helpers (same as live) ---
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
        return bool(self.orders or self.position)

    def _update_buffers(self):
        prev_close = self.closes[-1] if self.closes else self.data.close[0]
        self.ret_buffer.append((self.data.close[0] - prev_close) / prev_close if prev_close else 0.0)
        self.closes.append(self.data.close[0])

    def _pre_entry_gates(self):
        ok, why = self._risk_ready()
        if not ok:
            return False, why
        if self._in_cooldown():
            return False, f"cooldown: {self.cooldown_bars} bars"
        return True, None

    def _risk_ready(self):
        if len(self.ret_buffer) < self.min_warmup_bars:
            return False, f"warmup: need {self.min_warmup_bars} bars (have {len(self.ret_buffer)})"
        sigma = np.std(self.ret_buffer[-self.vol_window:]) if len(self.ret_buffer) >= self.vol_window else 0.0
        sigma_bps = 1e4 * sigma
        if sigma_bps < self.sigma_floor_bps:
            return False, f"sigma floor: {sigma_bps:.2f} < {self.sigma_floor_bps}"
        self._sigma_bps_latest = sigma_bps
        return True, None

    def _in_cooldown(self):
        return self.cooldown_bars > 0 and (len(self) - getattr(self, 'last_exit_bar', -10**9)) <= self.cooldown_bars

    def _post_entry_gates(self, sig, exp_bps, vote):
        if vote == 0:
            return False, "no signal"
        if sig == 0:
            return False, "EV gate"
        if self.p.enable_persistence and not self._pass_persistence(vote):
            return False, "persistence gate"
        if self.p.enable_quantile and not self._pass_quantile(exp_bps, vote):
            return False, "quantile gate"
        return True, None

    def _pass_persistence(self, vote):
        vb = self.vote_buffer
        return (len(vb) == vb.maxlen) and all(v == vote and v != 0 for v in vb)

    def _pass_quantile(self, exp_bps, vote):
        if len(self.exp_returns) < self.quantile_window:
            return True
        hist = np.array(self.exp_returns[-self.quantile_window:])
        long_thr  = np.quantile(hist, self.entry_quantile)
        short_thr = np.quantile(hist, 1 - self.entry_quantile)
        return not ((vote == 1 and exp_bps < long_thr) or (vote == -1 and exp_bps > short_thr))

    def _size_position(self, edge_like, sig):
        recent = self.edge_norms[-50:]
        ref = np.percentile(np.abs(recent), 90) if recent else 1e-4
        size_fac = min(abs(edge_like) / max(ref, 1e-4), 1.0) * self.max_position * sig
        cash = self.broker.getcash()
        raw = (cash * size_fac) / self.data.close[0]
        size = min(round(raw, 8), cash / self.data.close[0])
        return size if sig * size > 0 and abs(size) >= self.min_trade_size else None

    def _place_bracket(self, size, price):
        self.entry_order = self.buy(size=abs(size)) if size > 0 else self.sell(size=abs(size))
        self.orders = [self.entry_order]

    def _dbg(self, msg):
        if not self.cfg.get("backtest", {}).get("quiet", False):
            print(msg)

    def _log_bar_metrics(self, exp_bps, edge_like, vol, sig, thr):
        self.metrics_buffer.append({
            'datetime': self.data.datetime.datetime(0),
            'close': float(self.data.close[0]),
            'exp_r': exp_bps,
            'edge': edge_like,
            'volatility': vol,
            'signal': sig,
            'threshold': thr
        })

    def get_metrics(self):
        df = pd.DataFrame(self.metrics_buffer).rename(columns={'exp_r': 'exp_return', 'edge': 'edge_norm'})
        if 'position' not in df.columns:
            df['position'] = 0.0
        return df
