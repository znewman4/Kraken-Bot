# src/backtesting/runner.py



import backtrader as bt
import pandas as pd
from src.backtesting.strategies.strategy import KrakenStrategy
from src.backtesting.feeds import EngineeredData
from config_loader import load_config
from src.calibration import ensure_calibrators




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


    # in run_backtest(), after reading df:
    bt_cfg = config.get('backtest', {})
    start = bt_cfg.get('start_date')  # e.g., "2025-06-01"
    end   = bt_cfg.get('end_date')    # exclusive
    max_bars = config['backtest'].get('max_bars')


    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index < pd.to_datetime(end)]

    

    data = EngineeredData(dataname=df)
    cerebro.adddata(data)

    cerebro.broker.setcommission(
        commission=config['trading_logic']['fee_rate'],
        leverage=1.0
    )
    
    cerebro.broker.set_slippage_perc(
        perc=config['backtest'].get('slippage_perc', 0.0005),
        slip_open=True, slip_limit=True, slip_match=True
    )
    cerebro.broker.setcash(config['backtest']['cash'])

    # --- NEW: compute cost once ---
    fee_rate = config['trading_logic']['fee_rate']
    slip_rate = config['backtest'].get('slippage_perc', 0.0)
    cost_bps = (2 * fee_rate + slip_rate) * 1e4

    #Add calibrator
    ensure_calibrators(config)  # does nothing if disabled; fits if enabled+missing

    # pass to strategy
    cerebro.addstrategy(KrakenStrategy, config=config, cost_bps=cost_bps)

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


    return metrics_df, stats, cerebro





