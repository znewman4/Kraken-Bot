# src/backtesting/strategytest.py
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
        self.cfg    = self.p.config
        self.tl     = self.cfg['trading_logic']
        self.last_trade_info = {}

        # Load XGBoost models
        self.models        = {}
        self.feature_names = {}
        for h, path in self.tl['model_paths'].items():
            m = xgb.XGBRegressor()
            m.load_model(path)
            ih = int(h)
            self.models[ih]        = m
            self.feature_names[ih] = m.get_booster().feature_names

        # Trading parameters
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

        # Strategy state
        self.trade_log  = []
        self.entry_bar  = None
        self.orders     = []

        # Trackers and buffers
        self.closes        = []
        self.ret_buffer    = []
        self.exp_returns   = []
        self.edge_norms    = []
        self.thresholds    = []
        self.signals       = []
        self.signal_buffer = deque(maxlen=self.tl.get('persistence', 1))
        self.pnls          = []

        # Indicators
        self.atr = bt.indicators.ATR(self.data, period=self.tl.get('atr_period',14))
        self.sma = bt.indicators.SMA(self.data.close, period=self.cfg.get('features',{}).get('sma_window',20))
        self.rsi = bt.indicators.RSI(self.data.close, period=self.cfg.get('features',{}).get('rsi_window',14))
        self.tp_buffer  = deque(maxlen=self.vol_window)
        self.vol_buffer = deque(maxlen=self.vol_window)

        print(f"{datetime.utcnow().isoformat()} - KrakenStrategy initialized")

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return
        if order.status != order.Completed:
            return
        # Entry leg
        if order.parent is None:
            self.entry_bar = len(self)
            self.last_trade_info = {
                'entry_time':      self.data.datetime.datetime(0).isoformat(),
                'entry_price':     order.executed.price,
                'size':            order.executed.size,
                'predicted_exp_r': self.exp_returns[-1],
                'edge':            self.edge_norms[-1],
                'vol':             self.ret_buffer[-1] if self.ret_buffer else np.nan,
                'atr':             self.atr[0],
                'stop_dist':       self.stop_mult * self.atr[0],
                'tp_dist':         self.tp_mult   * self.atr[0],
                'bar_high':        self.data.high[0],
                'bar_low':         self.data.low[0],
                'bar_close':       self.data.close[0],
                'signal':          self.signals[-1],
                'exit_price':      None
            }
            print(f"{self.data.datetime.datetime(0).isoformat()} – ENTRY recorded: price={order.executed.price}, size={order.executed.size}")
            self.orders = []
            return
        # Exit leg
        exit_price = order.executed.price
        exit_size  = order.executed.size
        self.last_trade_info['exit_price'] = exit_price
        print(f"{self.data.datetime.datetime(0).isoformat()} – EXIT recorded: price={exit_price}, size={exit_size}")
        self.orders = []

    def notify_trade(self, trade):
        if trade.isclosed and hasattr(self, 'last_trade_info'):
            pnl = trade.pnlcomm
            entry_dt = pd.to_datetime(self.last_trade_info['entry_time'])
            hold_min = (self.data.datetime.datetime(0) - entry_dt).total_seconds() / 60
            info = self.last_trade_info
            self.trade_log.append(TradeLogEntry(
                entry_time      = info['entry_time'],
                exit_time       = self.data.datetime.datetime(0).isoformat(),
                entry_price     = info['entry_price'],
                exit_price      = info.get('exit_price', trade.price),
                position_size   = info['size'],
                predicted_exp_r = info['predicted_exp_r'],
                edge            = info['edge'],
                volatility      = info['vol'],
                net_pnl         = pnl,
                pnl_per_unit    = pnl / info['size'] if info['size'] else 0.0,
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
            self.last_trade_info = {}
            self.orders = []

    def next(self):
        dt = self.data.datetime.datetime(0)
        print(f"\n📌 BAR {len(self)} | Timestamp: {dt}")

        # 0) Clear stale orders
        if self.orders and not self.position:
            self.orders = []

        # 1) Timed exit
        if self.position and self.entry_bar is not None:
            bars_held = len(self) - self.entry_bar
            if bars_held >= self.max_hold_bars:
                print(f"🚪 Closing position after {bars_held} bars")
                for o in list(self.orders):
                    self.cancel(o)
                self.orders = []
                self.close()
                return

        # 2-9) VWAP, Predictions, Edge, Signal, Gating, Sizing
        tp = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3
        self.tp_buffer.append(tp)
        self.vol_buffer.append(self.data.volume[0])
        vwap = sum(p * v for p, v in zip(self.tp_buffer, self.vol_buffer)) / max(sum(self.vol_buffer), 1)
        print(f"📉 VWAP: {vwap:.4f}")

        preds = []
        for h, m in self.models.items():
            row = {}
            for f in self.feature_names[h]:
                lf = f.lower()
                if lf == 'sma':    row[f] = float(self.sma[0])
                elif lf == 'rsi':  row[f] = float(self.rsi[0])
                elif lf == 'vwap': row[f] = float(vwap)
                else:              row[f] = float(getattr(self.data, f)[0])
            pred = m.predict(pd.DataFrame([row]))[0]
            preds.append(pred)
            print(f"🔮 Pred h={h}: {pred:.6f}")
        weights = np.array(self.tl.get('horizon_weights', []), dtype=float)[:len(preds)]
        exp_r = float(np.dot(preds, weights)/weights.sum()) if weights.sum() else float(np.mean(preds))
        print(f"💡 Exp return: {exp_r:.6f}")

        if self.closes:
            r = (self.data.close[0] - self.closes[-1]) / self.closes[-1]
            self.ret_buffer.append(r)
        self.closes.append(self.data.close[0])
        vol = pd.Series(self.ret_buffer[-self.vol_window:]).std() if len(self.ret_buffer)>=self.vol_window else np.nan
        if vol>0:
            edge = exp_r/vol
            thr  = (self.fee_rate*self.threshold_mult)/vol
        else:
            edge, thr = 0.0, float('inf')
        sig = 1 if edge>=thr else -1 if edge<=-thr else 0
        print(f"📐 Edge={edge:.4f}, Thr={thr:.4f}, Sig={sig}")

        self.edge_norms.append(edge)
        self.thresholds.append(thr)
        self.exp_returns.append(exp_r)
        self.signals.append(sig)
        self.signal_buffer.append(sig)

        if sig==0:
            print("🚫 Sig=0: no trade")
            return
        if len(self.signal_buffer)<self.signal_buffer.maxlen or not all(s==sig for s in self.signal_buffer):
            print(f"🔄 Persist fail: {list(self.signal_buffer)}")
            return
        if len(self.exp_returns)>=self.quantile_window:
            hist = np.array(self.exp_returns[-self.quantile_window:])
            long_thr  = np.quantile(hist, self.entry_quantile)
            short_thr = np.quantile(hist, 1-self.entry_quantile)
            if sig==1 and exp_r<long_thr:
                print("📉 Long quantile fail")
                return
            if sig==-1 and exp_r>short_thr:
                print("📈 Short quantile fail")
                return

        # Position sizing
        if len(self.edge_norms)>=50:
            ref = np.percentile(np.abs(self.edge_norms[-50:]),90) or 1e-4
        else:
            ref = 1e-4
        size_fac = min(abs(edge)/ref,1.0)*self.max_position*sig
        cash     = self.broker.getcash()
        price    = self.data.close[0]
        raw_size = (cash*size_fac)/price
        size     = min(round(raw_size,8), cash/price)
        print(f"📏 size={size:.6f}")
        if abs(size)<self.min_trade_size:
            print("🚫 Size below min")
            return

        # 1) MARKET ENTRY on bar N
        if sig and not self.position and self.entry_bar is None:
            print(f"▶ ENTRY → {'BUY' if sig>0 else 'SELL'} @ {self.data.close[0]:.2f} on bar {len(self)}")
            o = self.buy(size=size) if sig>0 else self.sell(size=size)
            self.entry_bar = len(self)
            self.orders    = [o]
            return

        # 2) EXIT PLACEMENT on bar N+1
        if self.position and self.entry_bar is not None and len(self)>self.entry_bar:
            atr       = float(self.atr[0])
            stop_dist = self.stop_mult * atr
            tp_dist   = self.tp_mult   * atr
            tick      = self.tick_size
            if self.position.size>0:
                stop_price  = self.data.close[0] - stop_dist - tick
                limit_price = self.data.close[0] + tp_dist   + tick
                s_stop  = self.sell(exectype=bt.Order.Stop,  price=stop_price,  size=self.position.size)
                s_limit = self.sell(exectype=bt.Order.Limit, price=limit_price, size=self.position.size)
            else:
                stop_price  = self.data.close[0] + stop_dist + tick
                limit_price = self.data.close[0] - tp_dist   - tick
                s_stop  = self.buy(exectype=bt.Order.Stop,  price=stop_price,  size=abs(self.position.size))
                s_limit = self.buy(exectype=bt.Order.Limit, price=limit_price, size=abs(self.position.size))
            self.orders = [s_stop, s_limit]
            print(f"▶ EXIT ORDERS → Stop @ {stop_price:.2f}, Limit @ {limit_price:.2f} on bar {len(self)}")
            self.entry_bar = None
            return

        # 3) SKIP when in position or orders pending
        if self.position or self.orders:
            return

    def get_metrics(self):
        import pandas as pd
        n = len(self.closes)
        positions = getattr(self,'positions_log',[]) + [0.0]*(n-len(getattr(self,'positions_log',[])))
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
