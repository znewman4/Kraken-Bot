# src/backtesting/runner.py



import backtrader as bt
import pandas as pd
from src.backtesting.strategies.strategy import KrakenStrategy
from src.backtesting.feeds import make_dynamic_pandasdata
from config_loader import load_config



def run_backtest(config_path='config.yml'):

    print("Loading config from:", config_path)

    if isinstance(config_path, dict):
        config = config_path
        #print("Using provided config dict")
    else:
        from config_loader import load_config
        #print("Loading config from file:", config_path)
        config = load_config(config_path)

    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(False) 
    cerebro.broker.set_shortcash(True)

    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        timeframe=bt.TimeFrame.Minutes,   # we’re on minutes…
        compression=5,                     # …and each bar is 5-minutes long
        riskfreerate=0.0,                  # leave at zero unless you’ve got a RF curve                         
        )    

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
        df = df.tail(max_bars)



    DataCls = make_dynamic_pandasdata(df)
    data = DataCls(dataname=df)

    cerebro.adddata(data)    

    cerebro.addstrategy(KrakenStrategy, config=config)
    cerebro.broker.setcommission(
    commission=config['trading_logic']['fee_rate'],
    leverage=1.0
)

    cerebro.broker.set_slippage_perc(
        perc=config['backtest'].get('slippage_perc', 0.0005),
        slip_open=True, slip_limit=True, slip_match=True
    )
    cerebro.broker.setcash(config['backtest']['cash'])

    results = cerebro.run()
    strat = results[0]
    real_trade_pnls = strat.pnls      # list of net PnL from notify_trade()
    total_real_pnl  = sum(real_trade_pnls)

    sharpe_dict = strat.analyzers.sharpe.get_analysis()
    sharpe_val  = sharpe_dict.get('sharperatio', None)
    drawdown   = strat.analyzers.drawdown.get_analysis()
    trade_stats= strat.analyzers.trades.get_analysis()

    stats = {
        "sharpe": sharpe_val or 0.0,
        "drawdown": drawdown or 0.0,
        "trades": trade_stats or {},
        "real_pnl": total_real_pnl or 0.0,
    }

    strategy_instance = results[0]
    metrics_df = strategy_instance.get_metrics()

    trade_df = strat.get_trade_log_df()
    trade_df.to_csv("trade_log.csv", index=False) 

    import numpy as np
    # print("exp_r mean:", np.mean(strat.exp_returns))
    # print("exp_r std:", np.std(strat.exp_returns))
    # print("exp_r min:", np.min(strat.exp_returns))
    # print("exp_r max:", np.max(strat.exp_returns))

    if config.get("backtest", {}).get("plot", False):
        import matplotlib.pyplot as plt
        plt.hist(strat.exp_returns, bins=100)
        plt.title("Distribution of exp_r")
        plt.show()


    return metrics_df, stats, cerebro





