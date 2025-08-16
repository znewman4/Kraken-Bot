import backtrader as bt
import os
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd
from src.backtesting.strategies.strat_precomputed import KrakenStrategy
from src.backtesting.feeds import PrecomputedData
from config_loader import load_config

def run_backtest(config_or_path='config.yml'):
    cfg = config_or_path if isinstance(config_or_path, dict) else load_config(config_or_path)

    # --- CHANGED: prefer Parquet if provided in config ---
    pre_path = (
        cfg['data'].get('exp_return_parquet_path')  # NEW
        or cfg['data'].get('backtrader_data_path')
        or cfg['data']['feature_data_path']
    )

    # --- CHANGED: read parquet fast (memory-mapped) or fall back to CSV ---
    if str(pre_path).endswith('.parquet'):
        df = pd.read_parquet(pre_path, engine='pyarrow', memory_map=True)
        if df.index.name != 'time':
            # ensure Backtrader-friendly index
            if 'time' in df.columns:
                df = df.set_index('time')
            df.index.name = 'time'
    else:
        df = pd.read_csv(pre_path, parse_dates=['time']).set_index('time')
        df.index.name = 'time'

    # keep the guard (it’s helpful even with parquet)
    if 'exp_return' not in df.columns:
        raise ValueError(
            "Precomputed runner needs exp_return in data. "
            f"Got columns: {list(df.columns)[:12]}..."
        )

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

    cerebro.addstrategy(KrakenStrategy, config=cfg)

    cerebro.broker.setcommission(commission=cfg['trading_logic']['fee_rate'], leverage=1.0)
    cerebro.broker.set_slippage_perc(
        perc=cfg['backtest'].get('slippage_perc', 0.0005),
        slip_open=True, slip_limit=True, slip_match=True
    )
    cerebro.broker.setcash(cfg['backtest']['cash'])

    quiet = cfg.get('backtest', {}).get('quiet', True)  # default quiet for sweeps
    if quiet:
        with open(os.devnull, 'w') as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            results = cerebro.run()
    else:
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
