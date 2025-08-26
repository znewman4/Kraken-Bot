# src/backtesting/runner.py

import backtrader as bt
import pandas as pd
from src.backtesting.strategies.strategy import KrakenStrategy
from src.backtesting.feeds import EngineeredData
from config_loader import load_config


def _build_cerebro(config):
    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(False)
    cerebro.broker.set_shortcash(True)

    # use interval from config (was hardcoded 5)
    compression = int(config["exchange"].get("interval_minute", 5))
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        timeframe=bt.TimeFrame.Minutes,
        compression=compression,
        riskfreerate=0.0,
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    cerebro.addstrategy(KrakenStrategy, config=config)
    cerebro.broker.setcommission(commission=config['trading_logic']['fee_rate'], leverage=1.0)
    cerebro.broker.set_slippage_perc(
        perc=config['backtest'].get('slippage_perc', 0.0005),
        slip_open=True, slip_limit=True, slip_match=True
    )
    cerebro.broker.setcash(config['backtest']['cash'])
    return cerebro


def _run_core(df: pd.DataFrame, config):
    # Ensure Backtrader-friendly index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("runner: df.index must be a DatetimeIndex")
    if df.index.name != "time":
        df = df.copy()
        df.index.name = "time"

    data = EngineeredData(dataname=df)
    cerebro = _build_cerebro(config)
    cerebro.adddata(data)

    results = cerebro.run()
    strat = results[0]

    real_trade_pnls = getattr(strat, "pnls", [])
    total_real_pnl  = float(sum(real_trade_pnls)) if real_trade_pnls else 0.0

    sharpe_dict = strat.analyzers.sharpe.get_analysis() or {}
    drawdown    = strat.analyzers.drawdown.get_analysis() or {}
    trade_stats = strat.analyzers.trades.get_analysis() or {}

    stats = {
        "sharpe": sharpe_dict.get('sharperatio', 0.0) or 0.0,
        "drawdown": drawdown,
        "trades": trade_stats,
        "real_pnl": total_real_pnl,
    }

    metrics_df = strat.get_metrics()
    trade_df = strat.get_trade_log_df()
    trade_df.to_csv("trade_log.csv", index=False)

    if config.get("backtest", {}).get("plot", False):
        import matplotlib.pyplot as plt
        plt.hist(strat.exp_returns, bins=100)
        plt.title("Distribution of exp_r")
        plt.show()

    return metrics_df, stats, cerebro


def run_backtest_df(df: pd.DataFrame, config: dict):
    """
    Identical behavior to run_backtest, but uses the provided DataFrame (slice).
    Use from rolling_walkforward:
        df_test_bt = df_test.set_index("__dt__"); df_test_bt.index.name = "time"
        metrics, stats, _ = run_backtest_df(df_test_bt, cfg)
    """
    # respect optional max_bars for parity (trim at the end of the slice)
    max_bars = config.get('backtest', {}).get('max_bars')
    if max_bars:
        df = df.tail(max_bars)

    return _run_core(df, config)
