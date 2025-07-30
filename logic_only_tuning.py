import itertools
import pandas as pd
import numpy as np
from config_loader import load_config
from src.backtesting.runner_pnl_cv import run_backtest

# Load base config
cfg = load_config("config.yml")
tl  = cfg['trading_logic']

# Defaults from your config
default_qw     = tl['quantile_window']
default_eq     = tl['entry_quantile']
default_mts    = tl['min_trade_size']

# Two-parameter grid
hold_bars       = [29, 35, 48]               # your key horizons
threshold_mults = [0.01, 0.1, 0.3, 0.5, 1.0] # your main filter

grid = list(itertools.product(hold_bars, threshold_mults))

results = []
for i, (hbars, thr_mult) in enumerate(grid, 1):
    tl['max_hold_bars']  = hbars
    tl['threshold_mult'] = thr_mult
    # keep others fixed:
    tl['quantile_window'] = default_qw
    tl['entry_quantile']  = default_eq
    tl['min_trade_size']  = default_mts

    print(f"→ [{i}/{len(grid)}] hold={hbars}, thresh={thr_mult}")
    try:
        metrics, stats = run_backtest(cfg)
        trades   = stats.get('trades', {}).get('total',{}).get('total', 0)
        real_pnl = stats.get('real_pnl', 0.0)
        sharpe   = stats.get('sharpe', np.nan)
        print(f"    ✅ trades={trades}, PnL={real_pnl:.1f}, Sharpe={sharpe:.3f}")

        results.append({
            'hold_bars':    hbars,
            'threshold':    thr_mult,
            'n_trades':     trades,
            'total_pnl':    real_pnl,
            'sharpe':       sharpe,
        })

    except Exception as e:
        print(f"    ❌ error: {e}")

df = pd.DataFrame(results)
if df.empty:
    print("⚠️ No valid runs")
else:
    print("\n=== Top by Sharpe ===")
    print(df.sort_values('sharpe', ascending=False).head(10).to_string(index=False))
