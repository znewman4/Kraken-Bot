# hyperparameter_sweeper.py
import argparse, copy, yaml, tempfile, os, sys, itertools, multiprocessing
from pathlib import Path    
sys.path.append(str(Path(__file__).resolve().parents[3]))  # push repo root into path

import pandas as pd
import numpy as np
from config_loader import load_config
from joblib import Parallel, delayed
from src.backtesting.runners.runner_precomputed import run_backtest_df  # <- only need DF path

# ensure logs/ exists in project root
LOG_DIR = Path(__file__).resolve().parents[3] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# limit hidden threading inside numeric libs (prevents oversubscription)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Load base config (models already trained + paths set)
cfg_master = load_config("config.yml")
PARQ_PATH = cfg_master["data"]["exp_return_parquet_path"]  # pass path to workers

# --- your config test grid ---
quantile_windows      = [50, 100]                      
entry_quantiles       = [0.85, 0.9]                   
min_trade_sizes       = [0.01]                        
threshold_multipliers = [0.25, 0.3, 0.35, 0.4]         
max_hold_bars         = [75, 100]                  
stop_loss_mults       = [1.5, 2.0]               
take_profit_mults     = [4.0, 6.0]          

grid = list(itertools.product(
    quantile_windows,
    entry_quantiles,
    min_trade_sizes,
    threshold_multipliers,
    max_hold_bars,
    stop_loss_mults,
    take_profit_mults,
))

# ---- per-process lazy DataFrame cache ----
_G_DF = None
def _get_df():
    global _G_DF
    if _G_DF is None:
        df = pd.read_parquet(PARQ_PATH, engine="pyarrow", memory_map=True)
        df.index.name = "time"
        _G_DF = df
    return _G_DF

def run_single_config(i, qw, q, mts, thr_mult, hold, sl_mult, tp_mult, cfg_master):
    cfg = copy.deepcopy(cfg_master)
    tl = cfg['trading_logic']
    tl['quantile_window']        = qw
    tl['entry_quantile']         = q
    tl['min_trade_size']         = mts
    tl['threshold_mult']         = thr_mult
    tl['max_hold_bars']          = hold
    tl['stop_loss_atr_mult']     = sl_mult
    tl['take_profit_atr_mult']   = tp_mult
    # keep workers quiet
    cfg.setdefault('backtest', {})
    cfg['backtest'].setdefault('quiet', True)

    try:
        df = _get_df()  # each worker loads once, then reuses
        _, stats, _ = run_backtest_df(df, cfg)

        trades   = stats.get('trades', {})
        total_tr = trades.get('total', {}).get('total', 0) if isinstance(trades, dict) else 0
        real_pnl = stats.get('real_pnl', 0.0)
        sharpe   = stats.get('sharpe', np.nan)

        # return result dict (no global mutation)
        return {
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
        }
    except Exception as e:
        # surface errors but keep the sweep running
        return {'error': str(e), 'params': (qw, q, mts, thr_mult, hold, sl_mult, tp_mult)}

if __name__ == "__main__":
    # choose core count; start modest for the 4-config test
    n_jobs = multiprocessing.cpu_count()

    # run in parallel (loky backend uses processes by default)
    parallel_results = Parallel(
        n_jobs=n_jobs,
        prefer="processes",
        batch_size=8,   # group configs per dispatch; tune 4–20 later
        verbose=10
    )([
        delayed(run_single_config)(i, qw, q, mts, thr_mult, hold, sl_mult, tp_mult, cfg_master)
        for i, (qw, q, mts, thr_mult, hold, sl_mult, tp_mult) in enumerate(grid, 1)
    ])

    # collect
    results = [r for r in parallel_results if r and not r.get('error')]
    errors  = [r for r in parallel_results if r and r.get('error')]

    # --- Summaries ---
    df = pd.DataFrame(results)

    if df.empty:
        print("\n⚠️ No valid runs")
        if errors:
            print("Errors:", errors[:3], "…")
    else:
        print("\n=== 📈 Top Configs by Sharpe ===")
        print(df.sort_values(by="sharpe", ascending=False).head(10).to_string(index=False))

        print("\n=== 💰 Top Configs by Total PnL ===")
        print(df.sort_values(by="total_pnl", ascending=False).head(10).to_string(index=False))

        out_file = LOG_DIR / "gridsearch_results.csv"
        df.to_csv(out_file, index=False)
        print(f"\n✅ Full grid search results saved to {out_file}")
