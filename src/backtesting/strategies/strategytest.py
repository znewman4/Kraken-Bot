# src/backtesting/experimental_strategy.py
import backtrader as bt
import numpy as np
from collections import deque
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TradeLogEntry:
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_size: float
    predicted_exp_r: float
    edge: float
    volatility: float
    net_pnl: float
    pnl_per_unit: float
    hold_bars: int
    atr: float

class KrakenStrategy(bt.Strategy):
    params = (
        ('config', None),             # External config dict if provided
        ('forecast_horizon', 29),      # Default bars to hold before exit
        ('quantile_window', 100),      # Default rolling window for quantile gating
        ('entry_quantile', 0.8),       # Default quantile threshold for entries
    )

    def __init__(self, *args, **kwargs):
        # Allow Cerebro to inject config via params
        self.entry_order = None
        self.last_trade_info  = None    
        super().__init__(*args, **kwargs)
        # Load external config if available, else use params
        self.cfg = self.p.config or {}
        self.tl     = self.cfg['trading_logic']
        # Strategy parameters: prefer config values, fallback to params
        self.H  = int(self.tl.get('forecast_horizon', self.p.forecast_horizon))
        self.qw = int(self.tl.get('quantile_window',   self.p.quantile_window))
        self.eq = float(self.tl.get('entry_quantile',   self.p.entry_quantile))

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

        # state
        self.trade_log      = []
        self.entry_bar      = None
        self.bar_executed   = -1
        self.orders         = []

        # trackers
        self.closes        = []
        self.ret_buffer    = []
        self.exp_returns   = []
        self.edge_norms    = []
        self.thresholds    = []
        self.signals       = []
        self.signal_buffer = deque(maxlen=self.tl.get('persistence', 1))
        self.pnls          = []


        # ATR indicator for logging purposes
        self.atr = bt.indicators.ATR(self.data, period=14)

    def notify_order(self, order):
        if order is not self.entry_order:
            return
        if order.status == order.Completed and order.size:
            side = 'LONG' if order.isbuy() else 'SHORT'
            self.last_trade_info = {
                'entry_time':     self.data.datetime.datetime(0),
                'entry_price':    order.executed.price,
                'position_size':  order.executed.size,
                'predicted_exp_r': float(self.data.exp_return[0]),
                'edge':           self.current_edge,
                'volatility':     float(self.data.volatility_5[0]),
                'atr':            float(self.atr[0]),
                'entry_bar':      len(self),
            }
            self.entry_order = None
            print(f"▶ {side} entry at {order.executed.price:.4f}")

    def notify_trade(self, trade):
        # 1) Bail out if the trade isn't actually closed,
        #    or if we never set up entry info in notify_order.
        if not trade.isclosed or self.last_trade_info is None:
            return

        # 2) Safe to unpack the info dict now:
        info       = self.last_trade_info
        exit_time  = self.data.datetime.datetime(0)
        exit_price = trade.price
        net        = trade.pnlcomm
        size       = info['position_size']
        pnl_per    = net / size if size else 0.0
        hold       = len(self) - info['entry_bar']
        self.pnls.append(net)

        # 3) Log it
        self.trade_log.append(TradeLogEntry(
            entry_time      = info['entry_time'],
            exit_time       = exit_time,
            entry_price     = info['entry_price'],
            exit_price      = exit_price,
            position_size   = size,
            predicted_exp_r = info['predicted_exp_r'],
            edge            = info['edge'],
            volatility      = info['volatility'],
            net_pnl         = net,
            pnl_per_unit    = pnl_per,
            hold_bars       = hold,
            atr             = info['atr']
        ))
        print(f"◀ Trade closed | PnL={net:.4f}, bars held={hold}")

        # 4) Finally clear it so the next notify_trade won’t reuse stale info
        self.last_trade_info = None

    def _should_exit(self):
        if self.position and self.entry_bar is not None:
            bars_held = len(self) - self.entry_bar
            if bars_held >= self.H:
                self.close()
                print(f"Exiting after {bars_held} bars")
                self.entry_bar = None
                return True
        return False

    def _compute_edge_signal(self):
        exp_r = float(self.data.exp_return[0])
        recent = list(self.closes)
        vol = np.std(np.diff(recent)) if len(recent) >= 3 else np.nan
        edge = exp_r/vol if vol and vol > 0 else 0.0
        sig  = int(np.sign(edge))
        self.current_edge = edge
        return edge, sig

    def _passes_quantile(self, edge, sig):
        self.edge_norms.append(edge)
        if len(self.edge_norms) >= self.qw:
            hist = np.array(self.edge_norms[-self.qw:])
            long_thr  = np.quantile(hist, self.eq)
            short_thr = np.quantile(hist, 1 - self.eq)
            if (sig == 1 and edge < long_thr) or (sig == -1 and edge > short_thr):
                return False
        return True

    def _enter(self, sig):
        price = self.data.close[0]
        if sig == 1:
            self.entry_order = self.buy()
            print(f"Entry LONG | price={price:.4f}, edge={self.current_edge:.4f}")
        elif sig == -1:
            self.entry_order = self.sell()
            print(f"Entry SHORT | price={price:.4f}, edge={self.current_edge:.4f}")

    def _update_buffers(self):
        self.closes.append(self.data.close[0])

    def next(self):
        if self._should_exit():
            return
        if self.position:
            return

        edge, sig = self._compute_edge_signal()
        if not self._passes_quantile(edge, sig):
            return

        if sig != 0:
            self._enter(sig)
            self.entry_bar = len(self)

        self._update_buffers()


    def get_trade_log_df(self):
        import pandas as pd
        return pd.DataFrame([t.__dict__ for t in self.trade_log])


