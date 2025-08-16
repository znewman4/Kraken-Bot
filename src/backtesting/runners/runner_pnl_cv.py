import backtrader as bt
import pandas as pd
from src.backtesting.strategies.strategy_precomputed import KrakenStrategy
from src.backtesting.feeds import EngineeredData

def run_backtest(cfg):
    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(False)
    cerebro.broker.set_shortcash(True)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Minutes)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")




    df = pd.read_csv(cfg['data']['feature_data_path'], parse_dates=['time'])

    max_bars = cfg['backtest'].get('max_bars', None)
    if max_bars:
        df = df.tail(max_bars)
    df.set_index('time', inplace=True)

    data = EngineeredData(dataname=df)
    cerebro.adddata(data)

    cerebro.addstrategy(KrakenStrategy, config=cfg)
    cerebro.broker.setcommission(commission=cfg['trading_logic']['fee_rate'])
    cerebro.broker.setcash(cfg['backtest']['cash'])

    results = cerebro.run()

    strat = results[0]
    metrics = strat.get_metrics()
    real_trade_pnls = strat.pnls
    total_real_pnl = sum(real_trade_pnls)

    sharpe_val = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)

    # ✅ Extract analyzer data safely
    try:
        trade_analysis = strat.analyzers.trades.get_analysis()
    except Exception as e:
        print(f"⚠️ Could not extract TradeAnalyzer: {e}")
        trade_analysis = {}


    trade_df = strat.get_trade_log_df()
    trade_df.to_csv("trade_log.csv", index=False)

    return metrics, {
        "real_pnl": total_real_pnl,
        "sharpe": sharpe_val,
        "n_trades": len(real_trade_pnls),
        "trades": trade_analysis,
    }


