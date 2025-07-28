# src/backtesting/strategy3.py
import backtrader as bt
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import deque
from datetime import datetime
from dataclasses import dataclass

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

    # NEW fields:
    atr: float
    stop_dist: float
    tp_dist: float
    bar_high: float
    bar_low: float
    bar_close: float
    signal: int


    


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
        self.trade_log = []
        self.open_trades = {}  # map order.ref → info
        self.last_trade_info = None
        self.stop_mult = self.tl_cfg.get('stop_loss_atr_mult', 1.5)
        self.tp_mult   = self.tl_cfg.get('take_profit_atr_mult', 2.0)
        self.bar_executed = -1  # prevent same-bar double entries
        self.orders = []        # track open bracket orders



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
        if trade.isclosed and self.last_trade_info:
            pnl = trade.pnlcomm
            current_dt = self.datas[0].datetime.datetime(0)
            entry_dt = pd.to_datetime(self.last_trade_info["entry_time"])
            hold_minutes = (current_dt - entry_dt).total_seconds() / 60

            self.trade_log.append(TradeLogEntry(
                entry_time     = self.last_trade_info["entry_time"],
                exit_time      = current_dt.isoformat(),
                entry_price    = self.last_trade_info["entry_price"],
                exit_price     = trade.price,
                position_size  = self.last_trade_info["size"],
                predicted_exp_r= self.last_trade_info["predicted_exp_r"],
                edge           = self.last_trade_info["edge"],
                volatility     = self.last_trade_info["vol"],
                net_pnl        = pnl,
                pnl_per_unit   = pnl / self.last_trade_info["size"] if self.last_trade_info["size"] else 0.0,
                stop_hit       = pnl < 0,
                take_profit_hit= pnl > 0,
                hold_time_mins = hold_minutes,
                atr            = self.last_trade_info["atr"],
                stop_dist      = self.last_trade_info["stop_dist"],
                tp_dist        = self.last_trade_info["tp_dist"],
                bar_high       = self.last_trade_info["bar_high"],
                bar_low        = self.last_trade_info["bar_low"],
                bar_close      = self.last_trade_info["bar_close"],
                signal         = self.last_trade_info["signal"]
            ))
            self.pnls.append(pnl)
            print(f"🔍 TRADE CLOSED | Net PnL={pnl:.2f}")
            self.last_trade_info = None




    def notify_order(self, order):
        if order.status == bt.Order.Completed:
            if order.isbuy() or order.issell():
                self.last_trade_info = {
                    "entry_time": self.datas[0].datetime.datetime(0).isoformat(),
                    "entry_price": order.executed.price,
                    "size": order.executed.size,
                    "predicted_exp_r": self.exp_returns[-1] if self.exp_returns else 0.0,
                    "edge": self.edge_norms[-1] if self.edge_norms else 0.0,
                    "vol": self.ret_buffer[-1] if self.ret_buffer else 0.0,
                    "atr": self.atr[0],
                    "stop_dist": self.stop_mult * self.atr[0],
                    "tp_dist": self.tp_mult * self.atr[0],
                    "bar_high": self.data.high[0],
                    "bar_low": self.data.low[0],
                    "bar_close": self.data.close[0],
                    "signal": self.signals[-1] if self.signals else 0
                }

                side = 'BUY' if order.isbuy() else 'SELL'
                ts   = self.datas[0].datetime.datetime(0).isoformat()
                print(f"{ts} – {side} EXECUTED | ref={order.ref} | price={order.executed.price:.5f} | qty={order.executed.size:.6f}")

        elif order.status in (bt.Order.Canceled, bt.Order.Margin, bt.Order.Rejected, bt.Order.Completed):
            self.orders = []  # clear tracked bracket orders

            ts = self.datas[0].datetime.datetime(0).isoformat()
            status_name = order.getstatusname()
            print(f"{ts} – ⚠️ Order {status_name}")



    
    def next(self):
        # --- Step 0: Prevent Same-Bar Re-entry or Overlapping Brackets ---
        if len(self) == self.bar_executed:
            return  # already executed on this bar
        if self.position:
            return  # don't open new position if one is active
        if self.orders:
            return  # previous bracket still open

        # --- Step 1: VWAP Calculation ---
        tp = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3.0
        self.tp_buffer.append(tp)
        self.vol_buffer.append(self.data.volume[0])
        vwap = sum(p * v for p, v in zip(self.tp_buffer, self.vol_buffer)) / max(sum(self.vol_buffer), 1)

        # --- Step 2: Predict Expected Return ---
        preds = []
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

        w = np.array(weights[:len(preds)], dtype=float)
        exp_r = float(np.mean(preds)) if w.sum() == 0 else float(np.dot(preds, w) / w.sum())

        # --- Step 3: Compute Edge ---
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

        self.exp_returns.append(exp_r)
        self.edge_norms.append(edge)
        self.thresholds.append(thr)
        self.signals.append(sig)

        # --- Step 4: Gating based on signal persistence, time, position ---
        dt = self.datas[0].datetime.datetime(0)
        if not (8 <= dt.hour <= 20) or sig == 0:
            return
        if len(self.signal_buffer) < self.signal_buffer.maxlen or not all(s == sig for s in self.signal_buffer):
            return

        # --- Step 5: Position Sizing ---
        recent_edges = self.edge_norms[-50:]
        edge_norm_ref = np.percentile(recent_edges, 90) if len(recent_edges) >= 10 else 1e-4
        norm_edge = abs(edge) / edge_norm_ref
        norm_edge = min(norm_edge, 1.0)

        pos_fac = norm_edge * self.tl_cfg['max_position'] * sig
        cash_available = self.broker.getcash()
        price = self.data.close[0]
        size_req = (cash_available * abs(pos_fac)) / price
        max_affordable = cash_available / price
        size = min(size_req, max_affordable)
        size = round(size, 8)

        min_trade_size = self.tl_cfg.get("min_trade_size", 0.001)
        if size < min_trade_size:
            print(f"⚠️ Skipping micro-trade | size={size:.8f}, edge={edge:.5f}, norm_edge={norm_edge:.2f}")
            return

        # --- Step 6: Bracket Order with Tick Buffer ---
        stop_mult = self.tl_cfg.get('stop_loss_atr_mult', 5)
        tp_mult   = self.tl_cfg.get('take_profit_atr_mult', 7)
        tick_size = self.tl_cfg.get("tick_size", 0.01)

        atr_val = self.atr[0]
        stop_dist = stop_mult * atr_val
        tp_dist   = tp_mult   * atr_val

        if sig == 1:
            self.orders = self.buy_bracket(
                size       = size,
                price      = price,
                stopprice  = price - stop_dist - tick_size,
                limitprice = price + tp_dist + tick_size,
            )
        elif sig == -1:
            self.orders = self.sell_bracket(
                size       = size,
                price      = price,
                stopprice  = price + stop_dist + tick_size,
                limitprice = price - tp_dist - tick_size,
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
    
    def get_trade_log_df(self):
        import pandas as pd
        return pd.DataFrame([t.__dict__ for t in self.trade_log])
