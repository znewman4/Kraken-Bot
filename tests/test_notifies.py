# Put this file at: tests/test_notifies.py (pytest) or run as a script.
# It exercises your peeled-out `notifies.notify_order` and `notifies.notify_trade`
# WITHOUT needing a full Backtrader Cerebro run.

import types
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from datetime import datetime, timedelta
import math
import pytest

# --- Minimal fakes to stand in for Backtrader -------------------------------------------------

class FakeExecuted:
    def __init__(self, price=0.0, size=0.0):
        self.price = price
        self.size = size

class FakeOrder:
    # mimic bt.Order constants
    Submitted, Accepted, Completed, Canceled, Rejected = range(5)
    Market, Limit, Stop = range(3)

    def __init__(self, status, exectype, price, size, parent=None, info=None):
        self.status = status
        self.exectype = exectype
        self.executed = FakeExecuted(price, size)
        self.parent = parent
        self.info = info or {}

class FakeTrade:
    def __init__(self, isclosed, pnlcomm, price, baropen, barclose):
        self.isclosed = isclosed
        self.pnlcomm = pnlcomm
        self.price = price
        self.baropen = baropen
        self.barclose = barclose

class FakeLine:
    def __init__(self, value):
        self._v = value
    def __getitem__(self, idx):
        return self._v

class FakeDateTime:
    def __init__(self, t0):
        self.t = t0
    def datetime(self, _):
        return self.t

class FakeData:
    def __init__(self, t0, high, low, close):
        self.datetime = FakeDateTime(t0)
        self.high = FakeLine(high)
        self.low = FakeLine(low)
        self.close = FakeLine(close)

class FakeIndicator:
    def __init__(self, value):
        self._v = value
    def __getitem__(self, idx):
        return self._v

# --- Strategy stub holding exactly the fields your notifies.py uses ---------------------------

class StrategyStub:
    # Reuse your TradeLogEntry dataclass by constructing rows via __dict__ on append
    class TradeLogEntry:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def __init__(self):
        self._bar = 0
        t0 = datetime.utcnow()
        self.data = FakeData(t0, high=101.0, low=99.0, close=100.0)
        self.atr = FakeIndicator(2.0)
        self.tick_size = 0.01
        self.stop_mult = 4.0
        self.tp_mult = 6.0
        # strategy state used by notify functions
        self.entry_order = None
        self.stop_order = None
        self.limit_order = None
        self.children_armed = False
        self.entry_bar = None
        self.entry_fill_bar = None
        self.pending_exit_reason = None
        self.last_trade_info = {}
        self.trade_log = []
        self.pnls = []
        self.orders = []
        # buffers used for logging values
        self.exp_returns = [0.001]
        self.edge_norms = [0.1]
        self.ret_buffer = [0.0005]
        self.signals = [1]

    def __len__(self):
        return self._bar

    # helpers for tests
    def advance_bar(self, minutes=1):
        self._bar += 1
        self.data.datetime.t = self.data.datetime.t + timedelta(minutes=minutes)

# --- Import the functions under test ----------------------------------------------------------

# Adjust this import to match your project package name
from src.backtesting.engine import notifies

# --- Tests ------------------------------------------------------------------------------------

def test_entry_completion_arms_children():
    st = StrategyStub()
    st.advance_bar()  # bar 1

    # mark the order we will treat as entry
    entry = FakeOrder(FakeOrder.Completed, FakeOrder.Market, price=120000.0, size=0.02)
    st.entry_order = entry

    notifies.notify_order(st, entry)

    assert st.last_trade_info['entry_price'] == 120000.0
    assert st.children_armed is True
    # armed levels computed from ATR and multipliers
    assert math.isclose(st.armed_stop, 120000.0 - st.stop_mult*st.atr[0] - st.tick_size)
    assert math.isclose(st.armed_limit,120000.0 + st.tp_mult*st.atr[0]   + st.tick_size)


def test_exit_tp_tags_reason_and_clears_child_refs():
    st = StrategyStub()
    st.advance_bar()
    # prime as if after entry
    entry = FakeOrder(FakeOrder.Completed, FakeOrder.Market, 120000.0, 0.02)
    st.entry_order = entry
    notifies.notify_order(st, entry)

    # now simulate a TP fill (Limit)
    tp = FakeOrder(FakeOrder.Completed, FakeOrder.Limit, price=121200.0, size=-0.02, parent=entry)
    # track child refs to ensure they can be cleared
    st.limit_order = tp
    notifies.notify_order(st, tp)

    assert st.last_trade_info['exit_price'] == 121200.0
    assert st.last_trade_info['exit_reason'] == 'TP'
    assert st.limit_order is None  # cleared in helper


def test_exit_sl_tags_reason():
    st = StrategyStub()
    st.advance_bar()
    entry = FakeOrder(FakeOrder.Completed, FakeOrder.Market, 120000.0, 0.02)
    st.entry_order = entry
    notifies.notify_order(st, entry)

    sl = FakeOrder(FakeOrder.Completed, FakeOrder.Stop, price=118800.0, size=-0.02, parent=entry)
    st.stop_order = sl
    notifies.notify_order(st, sl)

    assert st.last_trade_info['exit_reason'] == 'SL'
    assert st.stop_order is None


def test_timed_manual_exit_path():
    st = StrategyStub()
    st.advance_bar()
    entry = FakeOrder(FakeOrder.Completed, FakeOrder.Market, 120000.0, 0.02)
    st.entry_order = entry
    notifies.notify_order(st, entry)

    # simulate you intentionally flattened via st.close(); parent=None and not the entry
    st.pending_exit_reason = 'Timed'
    manual_close = FakeOrder(FakeOrder.Completed, FakeOrder.Market, price=120500.0, size=-0.02, parent=None)
    notifies.notify_order(st, manual_close)

    assert st.last_trade_info['exit_reason'] == 'Timed'
    assert st.last_trade_info['exit_price'] == 120500.0


def test_notify_trade_writes_row_and_resets_state():
    st = StrategyStub()
    st.advance_bar()
    entry = FakeOrder(FakeOrder.Completed, FakeOrder.Market, 120000.0, 0.02)
    st.entry_order = entry
    notifies.notify_order(st, entry)
    # stage an exit via TP first to populate last_trade_info
    tp = FakeOrder(FakeOrder.Completed, FakeOrder.Limit, price=121200.0, size=-0.02, parent=entry)
    notifies.notify_order(st, tp)

    # closed trade comes in
    trade = FakeTrade(isclosed=True, pnlcomm=24.0, price=121100.0, baropen=st.entry_bar, barclose=len(st))
    notifies.notify_trade(st, trade)

    assert len(st.trade_log) == 1
    row = st.trade_log[0].__dict__
    assert row['take_profit_hit'] is True
    assert st.last_trade_info == {}
    assert st.children_armed is False
    assert st.entry_order is None and st.stop_order is None and st.limit_order is None


def test_notify_order_ignores_non_completed():
    st = StrategyStub()
    o1 = FakeOrder(FakeOrder.Submitted, FakeOrder.Market, 0, 0)
    o2 = FakeOrder(FakeOrder.Accepted,  FakeOrder.Market, 0, 0)
    o3 = FakeOrder(FakeOrder.Canceled,  FakeOrder.Market, 0, 0)
    for o in (o1,o2,o3):
        notifies.notify_order(st, o)
    assert st.last_trade_info == {}


def test_notify_trade_ignored_if_not_closed():
    st = StrategyStub()
    st.last_trade_info = {'entry_time': datetime.utcnow().isoformat(), 'entry_price': 100.0, 'size': 1.0}
    tr = FakeTrade(isclosed=False, pnlcomm=0.0, price=100.0, baropen=0, barclose=1)
    notifies.notify_trade(st, tr)
    assert len(st.trade_log) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
