import backtrader as bt
import pandas as pd
from src.backtesting.strategies.strategy_precomputed import KrakenStrategy
from src.backtesting.feeds import PrecomputedData
from config_loader import load_config

def run_backtest(config_or_path='config.yml'):
    cfg = config_or_path if isinstance(config_or_path, dict) else load_config(config_or_path)

    pre_path = cfg['data'].get('backtrader_data_path') or cfg['data']['feature_data_path']
    df = pd.read_csv(pre_path, parse_dates=['time']).set_index('time')

    max_bars = cfg['backtest'].get('max_bars')
    if max_bars:
        df = df.tail(max_bars)

    data = PrecomputedData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(False)
    cerebro.broker.set_shortcash(True)
    cerebro.adddata(data)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        timeframe=bt.TimeFrame.Minutes, compression=5, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    cerebro.addstrategy(
        KrakenStrategy,
        config=cfg,              # pass whole config as before
        # pre_col default is 'exp_return', but you can override via cfg['precomputed_col']
    )

    cerebro.broker.setcommission(commission=cfg['trading_logic']['fee_rate'], leverage=1.0)
    cerebro.broker.set_slippage_perc(
        perc=cfg['backtest'].get('slippage_perc', 0.0005),
        slip_open=True, slip_limit=True, slip_match=True
    )
    cerebro.broker.setcash(cfg['backtest']['cash'])

    results = cerebro.run()
    strat = results[0]

    # Quiet summaries only
    real_trade_pnls = getattr(strat, 'pnls', [])
    total_real_pnl  = float(sum(real_trade_pnls)) if real_trade_pnls else 0.0

    sharpe     = (strat.analyzers.sharpe.get_analysis() or {}).get('sharperatio', 0.0) or 0.0
    drawdown   = strat.analyzers.drawdown.get_analysis() or {}
    trade_stats= strat.analyzers.trades.get_analysis()  or {}

    stats = {
        "sharpe": sharpe,
        "drawdown": drawdown,
        "trades": trade_stats,
        "real_pnl": total_real_pnl,
    }

    metrics_df = strat.get_metrics()
    return metrics_df, stats, cerebro
