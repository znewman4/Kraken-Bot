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
    atr: float
    stop_dist: float
    tp_dist: float
    bar_high: float
    bar_low: float
    bar_close: float
    signal: int

class KrakenStrategy(bt.Strategy):
    params = (('config', None),)

    def __init__(self):
        # --- CONFIG & MODELS ---
        self.cfg    = self.p.config
        self.tl     = self.cfg['trading_logic']

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

        # indicators
        self.atr = bt.indicators.ATR(self.data, period=self.tl.get('atr_period',14))
        self.sma = bt.indicators.SMA(self.data.close,
                                     period=self.cfg.get('features',{}).get('sma_window',20))
        self.rsi = bt.indicators.RSI(self.data.close,
                                     period=self.cfg.get('features',{}).get('rsi_window',14))
        self.tp_buffer  = deque(maxlen=self.vol_window)
        self.vol_buffer = deque(maxlen=self.vol_window)

        print(f"{datetime.utcnow().isoformat()} - KrakenStrategy initialized")

    def notify_order(self, order):
        if order.status == bt.Order.Completed and (order.isbuy() or order.issell()):
            # record entry details
            self.entry_bar = len(self)
            self.last_trade_info = {
                'entry_time':     self.data.datetime.datetime(0).isoformat(),
                'entry_price':    order.executed.price,
                'size':           order.executed.size,
                'predicted_exp_r': self.exp_returns[-1],
                'edge':           self.edge_norms[-1],
                'vol':            self.ret_buffer[-1] if self.ret_buffer else np.nan,
                'atr':            self.atr[0],
                'stop_dist':      self.stop_mult * self.atr[0],
                'tp_dist':        self.tp_mult   * self.atr[0],
                'bar_high':       self.data.high[0],
                'bar_low':        self.data.low[0],
                'bar_close':      self.data.close[0],
                'signal':         self.signals[-1]
            }
            side = 'BUY' if order.isbuy() else 'SELL'
            print(f"{self.data.datetime.datetime(0).isoformat()} – {side} EXECUTED | price={order.executed.price:.5f} | qty={order.executed.size:.6f}")

    def notify_trade(self, trade):
        if trade.isclosed and hasattr(self, 'last_trade_info'):
            pnl = trade.pnlcomm
            entry_dt = pd.to_datetime(self.last_trade_info['entry_time'])
            hold_min = (self.data.datetime.datetime(0) - entry_dt).total_seconds()/60
            info = self.last_trade_info

            self.trade_log.append(TradeLogEntry(
                entry_time      = info['entry_time'],
                exit_time       = self.data.datetime.datetime(0).isoformat(),
                entry_price     = info['entry_price'],
                exit_price      = trade.price,
                position_size   = info['size'],
                predicted_exp_r = info['predicted_exp_r'],
                edge            = info['edge'],
                volatility      = info['vol'],
                net_pnl         = pnl,
                pnl_per_unit    = pnl/info['size'] if info['size'] else 0.0,
                stop_hit        = pnl < 0,
                take_profit_hit = pnl > 0,
                hold_time_mins  = hold_min,
                atr             = info['atr'],
                stop_dist       = info['stop_dist'],
                tp_dist         = info['tp_dist'],
                bar_high        = info['bar_high'],
                bar_low         = info['bar_low'],
                bar_close       = info['bar_close'],
                signal          = info['signal']
            ))
            self.pnls.append(pnl)
            print(f"🔍 TRADE CLOSED | Net PnL={pnl:.2f}")
            del self.last_trade_info
            self.orders = []

    def next(self):
        dt = self.data.datetime.datetime(0)
        print(f"BAR {len(self)} @ {dt.hour}: computing edge/threshold/sig")

        # 1) timed exit
        if self.position and self.entry_bar is not None:
            if len(self) - self.entry_bar >= self.max_hold_bars:
                print(f"{dt.isoformat()} – timed exit after {self.max_hold_bars} bars")
                for o in list(self.orders):
                    self.cancel(o)
                self.orders = []
                self.close()
                return

        # 2) skip if busy
        if self.position or self.orders:
            return

        # 3) VWAP
        tp = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3
        self.tp_buffer.append(tp)
        self.vol_buffer.append(self.data.volume[0])
        vwap = sum(p*v for p,v in zip(self.tp_buffer, self.vol_buffer)) / max(sum(self.vol_buffer),1)

        # 4) predictions
        preds = []
        for h,m in self.models.items():
            row = {}
            for f in self.feature_names[h]:
                lf = f.lower()
                if lf == 'sma':   row[f] = float(self.sma[0])
                elif lf == 'rsi': row[f] = float(self.rsi[0])
                elif lf == 'vwap':row[f] = float(vwap)
                else:             row[f] = float(getattr(self.data,f)[0])
            preds.append(m.predict(pd.DataFrame([row]))[0])
        w = np.array(self.tl.get('horizon_weights',[]),dtype=float)[:len(preds)]
        exp_r = float(np.dot(preds,w)/w.sum()) if w.sum() else float(np.mean(preds))

        # 5) edge & signal
        if self.closes:
            r = (self.data.close[0] - self.closes[-1]) / self.closes[-1]
            self.ret_buffer.append(r)
        self.closes.append(self.data.close[0])

        vol = pd.Series(self.ret_buffer[-self.vol_window:]).std() if len(self.ret_buffer)>=self.vol_window else np.nan
        if vol > 0:
            edge = exp_r / vol
            thr  = (self.fee_rate * self.threshold_mult) / vol
        else:
            edge, thr = 0.0, float('inf')

        sig = 1 if edge >= thr else (-1 if edge <= -thr else 0)
        self.edge_norms.append(edge)
        self.thresholds.append(thr)
        self.exp_returns.append(exp_r)
        self.signals.append(sig)
        self.signal_buffer.append(sig)

        # DEBUG summary
        print(f"[DBG] bar={len(self)} sig={sig} edge={edge:.4f} thr={thr:.4f} exp_r={exp_r:.6f}")

        # # 6) gating & quantile filter
        # if not (8 <= dt.hour <= 20):
        #     print("  ↳ DBG: blocked by time-of-day")
        #     return
        if sig == 0:
            print("  ↳ DBG: blocked because sig==0")
            return
        if len(self.signal_buffer) < self.signal_buffer.maxlen or not all(s == sig for s in self.signal_buffer):
            print("  ↳ DBG: blocked by persistence")
            return
        if len(self.exp_returns) >= self.quantile_window:
            hist = np.array(self.exp_returns[-self.quantile_window:])
            q = self.entry_quantile
            if sig == 1 and exp_r < np.quantile(hist, q):
                print(f"  ↳ DBG: blocked by quantile long ({exp_r:.4f} < Q{q:.2f}={np.quantile(hist,q):.4f})")
                return
            if sig == -1 and exp_r > np.quantile(hist, 1-q):
                print(f"  ↳ DBG: blocked by quantile short ({exp_r:.4f} > Q{1-q:.2f}={np.quantile(hist,1-q):.4f})")
                return

        # 7) position sizing
        ref       = np.percentile(self.edge_norms[-50:], 90) if len(self.edge_norms) >= 50 else 1e-4
        size_fac  = min(abs(edge)/ref, 1.0) * self.max_position * sig
        cash      = self.broker.getcash()
        price     = self.data.close[0]
        size      = min(round((cash * abs(size_fac)) / price, 8), cash/price)
        print(f"  ↳ DBG: sizing -> ref={ref:.4f}, size_fac={size_fac:.4f}, size={size:.6f}")
        if size < self.min_trade_size:
            print(f"  ↳ DBG: blocked by min_trade_size (size={size:.6f} < min={self.min_trade_size})")
            return

        # 8) entry bracket
        atr       = self.atr[0]
        stop_dist = self.stop_mult * atr
        tp_dist   = self.tp_mult   * atr
        tck       = self.tick_size

        print(f"  ↳ DBG: placing bracket sig={sig} size={size:.6f} price={price:.2f}")
        if sig == 1:
            self.orders = self.buy_bracket(
                size       = size,
                price      = price,
                stopprice  = price - stop_dist - tck,
                limitprice = price + tp_dist   + tck,
            )
        else:
            self.orders = self.sell_bracket(
                size       = size,
                price      = price,
                stopprice  = price + stop_dist + tck,
                limitprice = price - tp_dist   - tck,
            )
        print(f"  ↳ DBG: orders = {self.orders}")
        self.bar_executed = len(self)

    def get_metrics(self):
        import pandas as pd
        n = len(self.closes)
        positions = getattr(self, 'positions_log', []) + [0.0] * (n - len(getattr(self, 'positions_log', [])))
        return pd.DataFrame({
            'close':      self.closes,
            'exp_return': self.exp_returns,
            'edge_norm':  self.edge_norms,
            'threshold':  self.thresholds,
            'signal':     self.signals,
            'position':   positions,
        })

    def get_trade_log_df(self):
        return pd.DataFrame([t.__dict__ for t in self.trade_log])
