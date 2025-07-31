import backtrader as bt
import numpy as np
from collections import deque
from dataclasses import dataclass

@dataclass
class ExperimentTradeLogEntry:
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    net_pnl: float
    hold_time_bars: int
    predicted_exp_r: float
    edge: float
    volatility: float
    signal: int

class ExperimentStrategy(bt.Strategy):
    params = (
        ('config', None),
    )

    def __init__(self):
        cfg = self.p.config.get('experiment', {})
        # --- Experiment parameters ---
        self.horizon             = int(cfg.get('horizon', 29))
        self.use_persistence     = bool(cfg.get('use_persistence_filter', False))
        self.persistence         = int(cfg.get('signal_persistence', 2))
        self.use_quantile        = bool(cfg.get('use_quantile_gating', True))
        self.quantile_window     = int(cfg.get('quantile_window', 100))
        self.entry_quantile      = float(cfg.get('entry_quantile', 0.8))
        self.use_vol_filter      = bool(cfg.get('use_volatility_filter', True))
        # Precompute vol_threshold in config from your df: df['volatility_5'].quantile(0.69)
        self.vol_threshold       = float(cfg.get('vol_threshold', 0.0))
        self.use_edge_threshold  = bool(cfg.get('use_edge_threshold', False))
        self.edge_threshold      = float(cfg.get('edge_threshold', 0.95))

        # --- State trackers ---
        self.closes        = []
        self.exp_buffer    = []
        self.edge_buffer   = []
        self.sig_buffer    = deque(maxlen=self.persistence)
        self.in_trade      = False
        self.entry_bar     = None
        self.trade_log     = []

    def next(self):
        dt = self.data.datetime.datetime(0)

        # --- Exit Logic: hold to horizon bars ---
        if self.in_trade:
            bars_held = len(self) - self.entry_bar
            if bars_held >= self.horizon:
                exit_price = float(self.data.close[0])
                self.close()
                # record exit
                entry = self.trade_log[-1]
                entry.exit_time       = dt.isoformat()
                entry.exit_price      = exit_price
                entry.net_pnl         = (exit_price - entry.entry_price) / entry.entry_price * entry.signal
                entry.hold_time_bars  = bars_held
                self.in_trade = False
                return
            # still holding
            return

        # --- Compute signal & filters before entry ---
        exp_r = float(self.data.exp_return[0])
        # compute vol from last closes
        if len(self.closes) >= 3:
            vol = np.std(np.diff(self.closes[-20:]))
        else:
            vol = np.nan
        edge = exp_r / vol if vol > 0 else 0.0
        sig = 1 if edge > 0 else (-1 if edge < 0 else 0)

        # update history
        self.closes.append(float(self.data.close[0]))
        self.exp_buffer.append(exp_r)
        self.edge_buffer.append(edge)
        self.sig_buffer.append(sig)

        # --- Apply filters ---
        # Volatility regime
        if self.use_vol_filter and float(self.data.volatility_5[0]) > self.vol_threshold:
            return
        # Edge threshold
        if self.use_edge_threshold and abs(edge) < self.edge_threshold:
            return
        # Zero signal
        if sig == 0:
            return
        # Persistence
        if self.use_persistence and len(self.sig_buffer) == self.persistence:
            if not all(s == sig for s in self.sig_buffer):
                return
        # Quantile gating
        if self.use_quantile and len(self.edge_buffer) >= self.quantile_window:
            hist = np.array(self.edge_buffer[-self.quantile_window:])
            long_thr  = np.quantile(hist, self.entry_quantile)
            short_thr = np.quantile(hist, 1 - self.entry_quantile)
            if sig == 1 and edge < long_thr:
                return
            if sig == -1 and edge > short_thr:
                return

        # --- Enter trade at market ---
        size = self.broker.getcash() / float(self.data.close[0]) * 0.01  # example fixed 1% of cash; adjust as needed
        if sig == 1:
            self.buy(size=size)
        else:
            self.sell(size=size)

        # record entry
        entry = ExperimentTradeLogEntry(
            entry_time      = dt.isoformat(),
            exit_time       = None,
            entry_price     = float(self.data.close[0]),
            exit_price      = None,
            net_pnl         = None,
            hold_time_bars  = None,
            predicted_exp_r = exp_r,
            edge            = edge,
            volatility      = float(self.data.volatility_5[0]),
            signal          = sig
        )
        self.trade_log.append(entry)
        self.entry_bar = len(self)
        self.in_trade  = True

    def get_trade_log_df(self):
        import pandas as pd
        return pd.DataFrame([t.__dict__ for t in self.trade_log])
