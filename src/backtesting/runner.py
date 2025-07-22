# src/backtesting/runner.py

import backtrader as bt
import pandas as pd
import numpy as np
from src.backtesting.strategy3 import KrakenStrategy
from src.backtesting.feeds import EngineeredData
from config_loader import load_config


def run_backtest(config_path='config.yml'):

    print("Loading config from:", config_path)

    # ——— Load config & init Cerebro —————————————————————————————————
    config = load_config(config_path)
    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(True)

    # ——— Register analyzers up front —————————————————————————————
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio, _name="sharpe",
        timeframe=bt.TimeFrame.Minutes,
        compression=0
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown,      _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # ——— Load & slice data ———————————————————————————————————————
    df = pd.read_csv(
        config['data']['feature_data_path'],
        parse_dates=['time']
    )
    df.set_index('time', inplace=True)

    max_bars = config['backtest'].get('max_bars')
    if max_bars:
        df = df.iloc[:max_bars]

    data = EngineeredData(dataname=df)
    cerebro.adddata(data)

    # ——— Strategy, commission, slippage, cash —————————————————————
    cerebro.addstrategy(KrakenStrategy, config=config)
    cerebro.broker.setcommission(commission=config['trading_logic']['fee_rate'])
    cerebro.broker.set_slippage_perc(
        perc=config['backtest'].get('slippage_perc', 0.0005),
        slip_open=True, slip_limit=True, slip_match=True
    )
    cerebro.broker.setcash(config['trading_logic']['btc_stake'] * df['close'].iloc[0])

    # ——— Run backtest & pull analyzers ——————————————————————————
    results = cerebro.run()
    strat   = results[0]
    metrics = strat.get_metrics()

    sharpe_a   = strat.analyzers.sharpe.get_analysis()
    drawdown_a = strat.analyzers.drawdown.get_analysis()
    trades_a   = strat.analyzers.trades.get_analysis()

    # ——— Build stats dict with numeric Sharpe ——————————————————————
    stats = {
        "sharpe":   sharpe_a.get("sharperatio", np.nan),
        "drawdown": drawdown_a,
        "trades":   trades_a,
    }

    return metrics, stats, cerebro
