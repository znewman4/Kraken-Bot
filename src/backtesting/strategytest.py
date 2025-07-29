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
        print(f"ORDER NOTIFY: ref={order.ref}, status={order.getstatusname()}, "
            f"type={'BUY' if order.isbuy() else 'SELL'}, "
            f"created_price={order.created.price}, executed_price={order.executed.price}, size={order.created.size}")

        # Clear orders whenever they stop being active (not just completed):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.orders = []  # THIS IS CRITICAL TO AVOID PERPETUAL BLOCKS

        if order.status == order.Completed and (order.isbuy() or order.issell()):
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
        print(f"\n📌 BAR {len(self)} | Timestamp: {dt}")

        # 1) Timed exit logic
        if self.position and self.entry_bar is not None:
            bars_held = len(self) - self.entry_bar
            print(f"🕒 Checking timed exit: bars held = {bars_held}, max_hold_bars = {self.max_hold_bars}")
            if bars_held >= self.max_hold_bars:
                print(f"🚪 Exiting position due to timed exit after {bars_held} bars")
                for o in list(self.orders):
                    self.cancel(o)
                self.orders = []
                self.close()
                return

        # 2) Skip if busy
        if self.position:
            print("⛔️ Skipping next(), already in position.")
            return

        if self.orders:
            print("⛔️ Skipping next(), open orders pending.")
            return

        # 3) Compute VWAP
        tp = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3
        self.tp_buffer.append(tp)
        self.vol_buffer.append(self.data.volume[0])
        vwap = sum(p*v for p,v in zip(self.tp_buffer, self.vol_buffer)) / max(sum(self.vol_buffer), 1)
        print(f"📉 VWAP calculated: {vwap:.4f}")

        # 4) Predictions
        preds = []
        for h, m in self.models.items():
            row = {}
            for f in self.feature_names[h]:
                lf = f.lower()
                if lf == 'sma': row[f] = float(self.sma[0])
                elif lf == 'rsi': row[f] = float(self.rsi[0])
                elif lf == 'vwap': row[f] = float(vwap)
                else: row[f] = float(getattr(self.data, f)[0])
            pred = m.predict(pd.DataFrame([row]))[0]
            preds.append(pred)
            print(f"🔮 Prediction (horizon {h}): {pred:.6f}")

        weights = np.array(self.tl.get('horizon_weights', []), dtype=float)[:len(preds)]
        exp_r = float(np.dot(preds, weights)/weights.sum()) if weights.sum() else float(np.mean(preds))
        print(f"💡 Weighted expected return: {exp_r:.6f}")

        # 5) Edge & Signal
        if self.closes:
            r = (self.data.close[0] - self.closes[-1]) / self.closes[-1]
            self.ret_buffer.append(r)
        self.closes.append(self.data.close[0])

        vol = pd.Series(self.ret_buffer[-self.vol_window:]).std() if len(self.ret_buffer) >= self.vol_window else np.nan
        if vol > 0:
            edge = exp_r / vol
            thr = (self.fee_rate * self.threshold_mult) / vol
        else:
            edge, thr = 0.0, float('inf')

        sig = 1 if edge >= thr else (-1 if edge <= -thr else 0)

        print(f"📐 Calculated edge: {edge:.4f}, Threshold: {thr:.4f}, Signal: {sig}")

        self.edge_norms.append(edge)
        self.thresholds.append(thr)
        self.exp_returns.append(exp_r)
        self.signals.append(sig)
        self.signal_buffer.append(sig)

        # 6) Check signal
        if sig == 0:
            print("🚫 Trade blocked: Signal = 0")
            return

        # 7) Check persistence
        if len(self.signal_buffer) < self.signal_buffer.maxlen or not all(s == sig for s in self.signal_buffer):
            print(f"🔄 Trade blocked: Signal persistence failed ({list(self.signal_buffer)})")
            return

        # 8) Quantile filtering
        if len(self.exp_returns) >= self.quantile_window:
            hist = np.array(self.exp_returns[-self.quantile_window:])
            q = self.entry_quantile
            long_threshold = np.quantile(hist, q)
            short_threshold = np.quantile(hist, 1 - q)

            if sig == 1 and exp_r < long_threshold:
                print(f"📉 Trade blocked: Long quantile gating failed (exp_r={exp_r:.4f} < threshold={long_threshold:.4f})")
                return
            if sig == -1 and exp_r > short_threshold:
                print(f"📈 Trade blocked: Short quantile gating failed (exp_r={exp_r:.4f} > threshold={short_threshold:.4f})")
                return

        # 9) Position sizing
        ref = np.percentile(self.edge_norms[-50:], 90) if len(self.edge_norms) >= 50 else 1e-4
        size_fac = min(abs(edge)/ref, 1.0) * self.max_position * sig
        cash = self.broker.getcash()
        price = self.data.close[0]
        size = min(round((cash * abs(size_fac)) / price, 8), cash/price)

        print(f"📏 Position sizing: ref={ref:.4f}, size_fac={size_fac:.4f}, size={size:.6f}")

        if size < self.min_trade_size:
            print(f"🚫 Trade blocked: size ({size:.6f}) below min_trade_size ({self.min_trade_size})")
            return


        # 10) Bracket order placement
        atr = self.atr[0]
        stop_dist = self.stop_mult * atr
        tp_dist = self.tp_mult * atr
        tick = self.tick_size

        # Corrected Debug Diagnostics:
        if sig == 1:
            print(f"Bracket Prices (Long) | Market Entry: {price:.2f}, Stop Loss: {price - stop_dist - tick:.2f}, Take Profit: {price + tp_dist + tick:.2f}")
        else:
            print(f"Bracket Prices (Short) | Market Entry: {price:.2f}, Stop Loss: {price + stop_dist + tick:.2f}, Take Profit: {price - tp_dist - tick:.2f}")

        print(f"🎯 Placing MARKET bracket order: sig={sig}, size={size:.6f}, current_price={price:.2f}, stop_dist={stop_dist:.4f}, tp_dist={tp_dist:.4f}")

        # Corrected MARKET bracket orders (immediate execution at next open price)
        if sig == 1:
            self.orders = self.buy_bracket(
                size=size,
                stopprice=price - stop_dist - tick,
                limitprice=price + tp_dist + tick,
            )
        else:
            self.orders = self.sell_bracket(
                size=size,
                stopprice=price + stop_dist + tick,
                limitprice=price - tp_dist - tick,
            )

        self.bar_executed = len(self)
        print(f"✅ Bracket order placed successfully!")


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
