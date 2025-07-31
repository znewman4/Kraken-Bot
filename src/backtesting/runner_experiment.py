# runner_experiment.py
"""
Backtest runner that integrates RedisFeatureFeed and ExperimentStrategy.
1) Publishes historical bars + features + exp_returns into Redis.
2) Uses RedisFeatureFeed to consume the stream.
3) Runs the ExperimentStrategy in Backtrader.
"""
import sys, os
# ensure repo root is on PYTHONPATH to find config_loader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import backtrader as bt
from config_loader import load_config
from src.backtesting.redis_backtrader_integration import publish_from_csv, RedisFeatureFeed, redis_client, STREAM_KEY
from src.backtesting.strategy_experiment import ExperimentStrategy


def run_experiment():
    # Load configuration
    cfg = load_config('config.yml')

    # 2) Initialize Cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(False)        # no cheat-on-close
    cerebro.broker.set_shortcash(True)   # allow shorting

    # 3) Broker settings: commission, slippage, starting cash
    fee = cfg['trading_logic']['fee_rate']
    cerebro.broker.setcommission(commission=fee, leverage=1.0)
    slippage = cfg['backtest'].get('slippage_perc', 0.0005)
    cerebro.broker.set_slippage_perc(
        perc=slippage,
        slip_open=True, slip_limit=True, slip_match=True
    )
    cerebro.broker.setcash(cfg['backtest']['cash'])

    # 4) Add analyzers
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name='sharpe',
        timeframe=bt.TimeFrame.Minutes,
        compression=cfg['data'].get('compression', 5),
        riskfreerate=0.0
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown,    _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')


    # 5) Add data feed from Redis
    data = RedisFeatureFeed(
        host=cfg['stream']['host'],
        port=cfg['stream']['port'],
        db=cfg['stream']['db'],
        stream_key=cfg['stream']['feature_stream']
    )
    cerebro.adddata(data)

    # 1) Monkey-patch integration for diagnostics
    import src.backtesting.redis_backtrader_integration as redis_int
    _publish_count = {'count': 0}
    _orig_publish = redis_int.publish_bar
    def _debug_publish_bar(raw_bar):
        _publish_count['count'] += 1
        print(f"[Integration] publish_bar # {_publish_count['count']}: time={raw_bar['time']}")
        return _orig_publish(raw_bar)
    redis_int.publish_bar = _debug_publish_bar

    # 2) Populate Redis stream from CSV
    #    limit can be set in cfg['backtest']['max_bars'] or omitted for full history
    max_bars = cfg['backtest'].get('max_bars')
    publish_from_csv(limit=max_bars)
    # Diagnostic: print sample of the Redis stream to verify data ingestion
    print("=== Redis Stream Sample (first 5 entries) ===")
    try:
        # using redis_client and STREAM_KEY imported above
        sample = redis_client.xrange(STREAM_KEY, count=5)
        for entry in sample:
            print(entry)
    except Exception as e:
        print(f"Failed to fetch stream sample: {e}")

    # 6) Add the experiment strategy
    cerebro.addstrategy(ExperimentStrategy, config=cfg)

    # 7) Run backtest
    results = cerebro.run()
    strat = results[0]

    # 8) Extract results
    sharpe_val = strat.analyzers.sharpe.get_analysis().get('sharperatio')
    drawdown   = strat.analyzers.drawdown.get_analysis()
    trades     = strat.analyzers.trades.get_analysis()
    # Compute real PnL from the strategy's trade log DataFrame instead of strat.pnls
    trade_df   = strat.get_trade_log_df()
    real_pnl   = trade_df['net_pnl'].sum() if not trade_df.empty else 0.0
    stats = {
        'sharpe': sharpe_val,
        'drawdown': drawdown,
        'trades': trades,
        'real_pnl': real_pnl
    }
    print("Backtest statistics:", stats)

    # 9) Save trade log) Save trade log
    trade_df = strat.get_trade_log_df()
    trade_log_path = cfg['backtest'].get('trade_log_path', 'experiment_trade_log.csv')
    trade_df.to_csv(trade_log_path, index=False)
    print(f"Trade log written to {trade_log_path}")


if __name__ == '__main__':
    run_experiment()
