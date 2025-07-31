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
        cfg = self.p.config.get('trading_logic', {})

        # Hold and signal persistence
        self.horizon = int(cfg.get('max_hold_bars', 29))
        self.use_persistence = True
        self.persistence = int(cfg.get('persistence', 2))

        # Quantile gating (for edge)
        self.use_quantile = True
        self.quantile_window = int(cfg.get('quantile_window', 300))
        self.entry_quantile = float(cfg.get('entry_quantile', 0.6))

        # Volatility filter setup (dynamic sliding-window)
        self.use_vol_filter = True
        # Use same window length as quantile gating for consistency
        self.vol_buffer = deque(maxlen=self.quantile_window)
        self.vol_threshold = None  # will be recalculated each bar

        # Horizon models & weights
        model_paths = cfg.get('model_paths', {})
        self.horizon_list = sorted([int(h) for h in model_paths.keys()])
        self.horizon_weights = np.array(cfg.get('horizon_weights', []), dtype=float)
        if len(self.horizon_weights) != len(self.horizon_list):
            raise ValueError("Length of horizon_weights must match number of model_paths")
        self.horizon_weights = self.horizon_weights / self.horizon_weights.sum()

        # Edge threshold (if used)
        self.use_edge_threshold = False
        self.edge_threshold = float(cfg.get('threshold_mult', 0.5))

        # State trackers
        self.trade_log = []
        self.closes = deque(maxlen=self.quantile_window)
        self.exp_buffer = deque(maxlen=self.quantile_window)
        self.edge_buffer = deque(maxlen=self.quantile_window)
        self.sig_buffer = deque(maxlen=self.persistence)
        self.in_trade = False
        self.entry_bar = None

    def next(self):
        dt = self.datetime.datetime()

        # 0) Exit logic: close after holding for horizon bars
        if self.in_trade and (len(self) - self.entry_bar) >= self.horizon:
            self.close()
            last = self.trade_log[-1]
            last.exit_time      = dt.isoformat()
            last.exit_price     = float(self.data.close[0])
            last.hold_time_bars = self.horizon
            last.net_pnl        = last.signal * (last.exit_price - last.entry_price)
            self.in_trade       = False

        # 1) Collect single-horizon predictions
        preds = [
            float(getattr(self.data, f'exp_return_{h}')[0])
            for h in self.horizon_list
        ]
        # 2) Combine via weighted average
        exp_r = float(np.dot(preds, self.horizon_weights))

        # 3) Compute volatility
        vol = float(self.data.volatility_5[0])
        self.vol_buffer.append(vol)
        # Dynamic volatility threshold (69th percentile of recent vol)
        if len(self.vol_buffer) >= self.quantile_window:
            clean_vols = [v for v in self.vol_buffer if v > 0]
            self.vol_threshold = float(np.quantile(clean_vols, 0.69)) if clean_vols else float('inf')
            # debug print when threshold updates each bar (optional)
            print(f"{dt.isoformat()} | Dynamic vol_threshold={self.vol_threshold:.6e}")
        else:
            # not enough data yet, set threshold high to avoid blocking
            self.vol_threshold = float('inf')

        # 4) Compute edge & signal
        edge = exp_r / vol if vol > 0 else 0.0
        sig = 1 if edge > 0 else (-1 if edge < 0 else 0)

        # Debug output
        print(f"{dt.isoformat()} | preds={preds} | weights={self.horizon_weights.tolist()}"
              f" | exp_r={exp_r:.6f} | vol={vol:.6e} | edge={edge:.6f}")

        # Update history buffers
        self.closes.append(float(self.data.close[0]))
        self.exp_buffer.append(exp_r)
        self.edge_buffer.append(edge)
        self.sig_buffer.append(sig)

        # --- Apply filters ---
        # Volatility regime (dynamic)
        if self.use_vol_filter and vol > self.vol_threshold:
            return
        # Edge threshold
        if self.use_edge_threshold and abs(edge) < self.edge_threshold:
            return
        # Zero signal
        if sig == 0:
            return
        # Persistence filter
        if self.use_persistence and len(self.sig_buffer) == self.persistence:
            if not all(s == sig for s in self.sig_buffer):
                return
        # Quantile gating on edge (dynamic)
        if self.use_quantile and len(self.edge_buffer) >= self.quantile_window:
            hist = np.array(self.edge_buffer)
            long_thr = np.quantile(hist, self.entry_quantile)
            short_thr = np.quantile(hist, 1 - self.entry_quantile)
            if sig == 1 and edge < long_thr:
                return
            if sig == -1 and edge > short_thr:
                return

        # --- Enter trade ---
        size = self.broker.getcash() / float(self.data.close[0]) * 0.01
        if sig == 1:
            self.buy(size=size)
        elif sig == -1:
            self.sell(size=size)

        # Log trade entry
        entry = ExperimentTradeLogEntry(
            entry_time      = dt.isoformat(),
            exit_time       = None,
            entry_price     = float(self.data.close[0]),
            exit_price      = None,
            net_pnl         = None,
            hold_time_bars  = None,
            predicted_exp_r = exp_r,
            edge            = edge,
            volatility      = vol,
            signal          = sig
        )
        self.trade_log.append(entry)
        self.entry_bar = len(self)
        self.in_trade = True

    def get_trade_log_df(self):
        import pandas as pd
        return pd.DataFrame([t.__dict__ for t in self.trade_log])
