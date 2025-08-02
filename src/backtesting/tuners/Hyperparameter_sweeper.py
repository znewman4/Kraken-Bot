import itertools
import pandas as pd
import numpy as np
from config_loader import load_config
from src.backtesting.runners.runner_pnl_cv import run_backtest

# Load base config (models already trained + paths set)
cfg = load_config("config.yml")

quantile_windows      = [29, 50, 100, 200]
entry_quantiles       = [0.80, 0.90, 0.95, 0.99]
threshold_multipliers = [0.01, 0.1, 0.3, 0.5, 1.0]
min_trade_sizes       = [0.01, 0.02, 0.05]

# Build the full grid of combinations
grid = list(itertools.product(
    quantile_windows,
    entry_quantiles,
    min_trade_sizes,
    threshold_multipliers,
))

results = []

for i, (qw, q, mts, thr_mult) in enumerate(grid, 1):
    # Inject into strategy3 via config
    cfg['trading_logic']['quantile_window']  = qw
    cfg['trading_logic']['entry_quantile']   = q
    cfg['trading_logic']['min_trade_size']   = mts
    cfg['trading_logic']['threshold_mult']   = thr_mult

    print(f"→ [{i}/{len(grid)}] quantile_window={qw}, entry_quantile={q}, "
          f"min_trade_size={mts}, threshold_mult={thr_mult}")
    try:
        # Run the backtest with the current parameter combo
        metrics, stats = run_backtest(cfg)

        # Extract stats
        trades   = stats.get('trades', {})
        total_tr = trades.get('total', {}).get('total', 0) if isinstance(trades, dict) else 0
        real_pnl = stats.get('real_pnl', 0.0)
        sharpe   = stats.get('sharpe', np.nan)

        print(f"    ✅ Trades={total_tr}, PnL={real_pnl:.2f}, Sharpe={sharpe:.4f}")

        results.append({
            'quantile_window': qw,
            'entry_quantile':  q,
            'min_trade_size':  mts,
            'threshold_mult':  thr_mult,
            'n_trades':        total_tr,
            'total_pnl':       real_pnl,
            'sharpe':          sharpe,
        })

    except Exception as e:
        print(f"    ❌ error: {e}")

# Summarize results
df = pd.DataFrame(results)

if df.empty:
    print("\n⚠️ No valid runs")
else:
    print("\n=== Top Configs by Total PnL ===")
    print(df.sort_values(by="total_pnl", ascending=False).to_string(index=False))
