# src/backtesting/runner.py



import backtrader as bt
import pandas as pd
from src.backtesting.strategy2 import KrakenStrategy
from src.backtesting.feeds import EngineeredData
from config_loader import load_config



def run_backtest(config_path='config.yml'):

    print("Loading config from:", config_path)

    config = load_config(config_path)
    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(True) 

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown,    _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    df = pd.read_csv(
        config['data']['feature_data_path'],
        parse_dates=['time']
    )
    df.set_index('time', inplace=True)

    # now optionally slice to first N bars:
    max_bars = config['backtest'].get('max_bars')
    if max_bars:
        df = df.iloc[:max_bars]

    data = EngineeredData(dataname=df)
    cerebro.adddata(data)

    cerebro.addstrategy(KrakenStrategy, config=config)
    cerebro.broker.setcommission(commission=config['trading_logic']['fee_rate'])
    cerebro.broker.set_slippage_perc(
        perc=config['backtest'].get('slippage_perc', 0.0005),
        slip_open=True, slip_limit=True, slip_match=True
    )
    cerebro.broker.setcash(config['trading_logic']['btc_stake'] * df['close'].iloc[0])

    results = cerebro.run()
    strat = results[0]
    metrics = strat.get_metrics()

    #sharpe = cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")

    sharpe = cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        timeframe = bt.TimeFrame.Minutes,    # use minute bars
        compression = 0,                    # group every 15 bars → 15 min
)    
    drawdown   = strat.analyzers.drawdown.get_analysis()
    trade_stats= strat.analyzers.trades.get_analysis()

    stats = {
        "sharpe": sharpe,
        "drawdown": drawdown,
        "trades": trade_stats,
    }


    return metrics, stats, cerebro
