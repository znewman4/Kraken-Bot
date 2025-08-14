# src/backtesting/engine/notifies.py
import backtrader as bt
from datetime import datetime
import pandas as pd  # used for hold-time calc in notify_trade

def notify_order(st, order):
    """
    Exact logic from your Strategy.notify_order, but as a helper.
    Pass in the Strategy instance as `st` and the Backtrader `order`.
    """
    # No info on fills yet
    if order.status in (order.Submitted, order.Accepted):
        return

    # Optional visibility (OCO cancels land here)
    if order.status == order.Canceled:
        # print(f"{st.data.datetime.datetime(0).isoformat()} – ❌ Canceled: {order.info.get('tag','')}")
        return

    if order.status != order.Completed:
        return

    now = st.data.datetime.datetime(0).isoformat()

    # ---------- ENTRY COMPLETED ----------
    if st.entry_order is order:
        px   = float(order.executed.price)
        sz   = float(order.executed.size)
        atr  = float(st.atr[0])
        tick = float(st.tick_size)

        st.entry_bar      = len(st)
        st.entry_fill_bar = len(st)
        st.last_trade_info = {
            'entry_time':      now,
            'entry_price':     px,
            'size':            sz,
            'predicted_exp_r': float(st.exp_returns[-1]) if st.exp_returns else float('nan'),
            'edge':            float(st.edge_norms[-1]) if st.edge_norms else float('nan'),
            'volatility':      float(st.ret_buffer[-1]) if st.ret_buffer else float('nan'),
            'atr':             atr,
            'stop_dist':       float(st.stop_mult * atr),
            'tp_dist':         float(st.tp_mult   * atr),
            'bar_high':        float(st.data.high[0]),
            'bar_low':         float(st.data.low[0]),
            'bar_close':       float(st.data.close[0]),
            'signal':          int(st.signals[-1]) if st.signals else 0,
            'entry_bar':       st.entry_bar,
            'exit_bar':        None,
            'bars_held':       None,
            'exit_reason':     None,
        }

        side = 1 if sz > 0 else -1
        if side > 0:
            st.armed_stop  = px - st.stop_mult * atr - tick
            st.armed_limit = px + st.tp_mult   * atr + tick
        else:
            st.armed_stop  = px + st.stop_mult * atr + tick
            st.armed_limit = px - st.tp_mult   * atr - tick

        st.children_armed = True
        st.entry_order    = order
        print(f"{now} – ▶ ENTRY filled: price={px:.6f}, size={sz:.6f} (children will arm next bar)")
        return

    # ---------- EXIT (CHILD) COMPLETED ----------
    ex_price = float(order.executed.price)
    ex_size  = float(order.executed.size)
    reason   = "TP" if order.exectype == bt.Order.Limit else ("SL" if order.exectype == bt.Order.Stop else "Other")

    if st.last_trade_info:
        st.last_trade_info['exit_price']  = ex_price
        st.last_trade_info['exit_reason'] = reason
        st.last_trade_info['exit_bar']    = len(st)

    # clear refs so _in_flight() reflects reality
    if st.limit_order is order:
        st.limit_order = None
    if st.stop_order is order:
        st.stop_order = None

    print(f"{now} – ◀ EXIT filled ({reason}): price={ex_price:.6f}, size={ex_size:.6f}")

    # ---------- TIMED / MANUAL EXIT COMPLETED ----------
    # If we intentionally flattened (e.g., timed exit via st.close()), record it here.
    # Guard against misclassifying the entry: it must not be the tracked entry.
    if (
        st.pending_exit_reason
        and st.last_trade_info
        and (st.entry_order is not order)   # ensure this isn't the entry's Completed
        and order.parent is None            # close() orders have no parent
    ):
        ex_price = float(order.executed.price)
        ex_size  = float(order.executed.size)

        st.last_trade_info['exit_price']  = ex_price
        st.last_trade_info['exit_reason'] = st.pending_exit_reason  # e.g., "Timed"
        st.last_trade_info['exit_bar']    = len(st)

        # clear reason flag until next intentional flatten
        st.pending_exit_reason = None

        print(f"{now} – ◀ EXIT filled ({st.last_trade_info['exit_reason']}): "
              f"price={ex_price:.6f}, size={ex_size:.6f}")

        return


def notify_trade(st, trade):
    """
    Exact logic from your Strategy.notify_trade, but as a helper.
    """
    if not trade.isclosed or not st.last_trade_info:
        return

    info = st.last_trade_info
    pnl  = float(trade.pnlcomm)

    # feed-time timestamp for consistency
    exit_time = st.data.datetime.datetime(0).isoformat()
    entry_dt  = pd.to_datetime(info['entry_time'])
    hold_min  = (st.data.datetime.datetime(0) - entry_dt).total_seconds() / 60.0

    open_bar  = info.get('entry_bar', trade.baropen)
    close_bar = info.get('exit_bar', trade.barclose)
    bars_held = (close_bar - open_bar) if (close_bar is not None and open_bar is not None) else (trade.barclose - trade.baropen)

    # robust exit price fallback
    _exit_px = info.get('exit_price')
    _exit_px = _exit_px if _exit_px is not None else trade.price

    st.trade_log.append(type(st).TradeLogEntry(  # uses the dataclass on your class
        entry_time      = info['entry_time'],
        exit_time       = exit_time,
        entry_price     = float(info['entry_price']),
        exit_price      = float(_exit_px),
        position_size   = float(info['size']),
        predicted_exp_r = float(info['predicted_exp_r']),
        edge            = float(info['edge']),
        volatility      = float(info['volatility']),
        net_pnl         = pnl,
        pnl_per_unit    = pnl / float(info['size']) if info['size'] else 0.0,
        stop_hit        = info.get('exit_reason') == 'SL',
        take_profit_hit = info.get('exit_reason') == 'TP',
        hold_time_mins  = hold_min,
        atr             = float(info['atr']),
        stop_dist       = float(info['stop_dist']),
        tp_dist         = float(info['tp_dist']),
        bar_high        = float(info['bar_high']),
        bar_low         = float(info['bar_low']),
        bar_close       = float(info['bar_close']),
        signal          = int(info['signal']),
        entry_bar       = int(open_bar) if open_bar is not None else int(trade.baropen),
        exit_bar        = int(close_bar) if close_bar is not None else int(trade.barclose),
        bars_held       = int(bars_held),
    ))
    st.pnls.append(pnl)
    print(f"🔍 TRADE CLOSED | {info.get('exit_reason','?')} | Net PnL={pnl:.2f} | Bars held={int(bars_held)}")

    # reset per-trade state
    st.last_trade_info = {}
    st.children_armed  = False
    st.entry_order     = None
    st.stop_order      = None
    st.limit_order     = None
    st.orders          = []
