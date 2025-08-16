import argparse, copy, yaml, tempfile, os, sys, itertools
from pathlib import Path    
sys.path.append(str(Path(__file__).resolve().parents[3]))  # push repo root into path

import pandas as pd
import numpy as np
from config_loader import load_config
from src.backtesting.runners.runner_precomputed import run_backtest   # 🔑 use real backtester

# Load base config (models already trained + paths set)
cfg_master = load_config("config.yml")

quantile_windows      = [29, 50, 100, 200]
entry_quantiles       = [0.80, 0.90, 0.95, 0.99]
threshold_multipliers = [0.01, 0.1, 0.3, 0.5, 1.0]
min_trade_sizes       = [0.01, 0.02, 0.05]

# new knobs
max_hold_bars         = [20, 29, 50, 100]
stop_loss_mults       = [1.5, 2.0, 3.0]
take_profit_mults     = [3.0, 4.0, 6.0]

grid = list(itertools.product(
    quantile_windows,
    entry_quantiles,
    min_trade_sizes,
    threshold_multipliers,
    max_hold_bars,
    stop_loss_mults,
    take_profit_mults,
))

results = []

for i, (qw, q, mts, thr_mult, hold, sl_mult, tp_mult) in enumerate(grid, 1):
    cfg = copy.deepcopy(cfg_master)
    cfg['trading_logic']['quantile_window']   = qw
    cfg['trading_logic']['entry_quantile']    = q
    cfg['trading_logic']['min_trade_size']    = mts
    cfg['trading_logic']['threshold_mult']    = thr_mult
    cfg['trading_logic']['max_hold_bars']     = hold
    cfg['trading_logic']['stop_loss_atr_mult']   = sl_mult
    cfg['trading_logic']['take_profit_atr_mult'] = tp_mult

    print(f"→ [{i}/{len(grid)}] "
          f"qw={qw}, q={q}, mts={mts}, thr_mult={thr_mult}, "
          f"hold={hold}, sl_mult={sl_mult}, tp_mult={tp_mult}")

    try:
        _, stats, _ = run_backtest(cfg)

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
            'max_hold_bars':   hold,
            'stop_loss_mult':  sl_mult,
            'take_profit_mult': tp_mult,
            'n_trades':        total_tr,
            'total_pnl':       real_pnl,
            'sharpe':          sharpe,
        })
    except Exception as e:
        print(f"    ❌ error: {e}")

# --- Summaries ---
df = pd.DataFrame(results)

if df.empty:
    print("\n⚠️ No valid runs")
else:
    print("\n=== 📈 Top Configs by Sharpe ===")
    print(df.sort_values(by="sharpe", ascending=False).head(10).to_string(index=False))

    print("\n=== 💰 Top Configs by Total PnL ===")
    print(df.sort_values(by="total_pnl", ascending=False).head(10).to_string(index=False))

    df.to_csv("gridsearch_results.csv", index=False)
    print("\n✅ Full grid search results saved to gridsearch_results.csv")
