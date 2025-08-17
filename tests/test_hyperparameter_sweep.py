#test_hyperparameter_sweep.py

# tests/test_sweep_small.py
# Sanity-check hyperparameter sweep:
# - serial vs parallel parity (same trades/PnL/Sharpe)
# - quick timing
# Run from repo root:
#   python -m tests.test_sweep_small
# or with pytest:
#   pytest -q tests/test_sweep_small.py

import os, sys, copy, time, itertools, multiprocessing, numpy as np, pandas as pd
from pathlib import Path

# Make repo root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config_loader import load_config
from joblib import Parallel, delayed
from src.backtesting.runners.runner_precomputed import run_backtest_df

# ------------ small 4-config grid (fast) ------------
GRID = list(itertools.product(
    [29, 50],        # quantile_windows (2)
    [0.80, 0.90],    # entry_quantiles (2)
    [0.01],          # min_trade_sizes (1)
    [0.1],           # threshold_multipliers (1)
    [29],            # max_hold_bars (1)
    [2.0],           # stop_loss_mults (1)
    [4.0],           # take_profit_mults (1)
))
# Total: 4 configs

# ------------ per-process lazy Parquet preload ------------
_G_DF = None
_PARQ_PATH = None

def _get_df():
    """Load Parquet once per process; memory-mapped for speed."""
    global _G_DF
    if _G_DF is None:
        df = pd.read_parquet(_PARQ_PATH, engine="pyarrow", memory_map=True)
        df.index.name = "time"
        _G_DF = df
    return _G_DF

def _run_single(params, cfg_master):
    qw, q, mts, thr_mult, hold, sl_mult, tp_mult = params
    cfg = copy.deepcopy(cfg_master)
    tl = cfg["trading_logic"]
    tl["quantile_window"]      = qw
    tl["entry_quantile"]       = q
    tl["min_trade_size"]       = mts
    tl["threshold_mult"]       = thr_mult
    tl["max_hold_bars"]        = hold
    tl["stop_loss_atr_mult"]   = sl_mult
    tl["take_profit_atr_mult"] = tp_mult
    cfg.setdefault("backtest", {}).setdefault("quiet", True)

    df = _get_df()
    _, stats, _ = run_backtest_df(df, cfg)

    trades   = stats.get("trades", {})
    total_tr = trades.get("total", {}).get("total", 0) if isinstance(trades, dict) else 0
    return {
        "quantile_window":  qw,
        "entry_quantile":   q,
        "min_trade_size":   mts,
        "threshold_mult":   thr_mult,
        "max_hold_bars":    hold,
        "stop_loss_mult":   sl_mult,
        "take_profit_mult": tp_mult,
        "n_trades":         total_tr,
        "total_pnl":        float(stats.get("real_pnl", 0.0)),
        "sharpe":           float(stats.get("sharpe", np.nan)),
    }

def _stable(df):
    cols = ["quantile_window","entry_quantile","min_trade_size","threshold_mult",
            "max_hold_bars","stop_loss_mult","take_profit_mult"]
    return df.sort_values(cols).reset_index(drop=True)

def main():
    # avoid hidden thread oversubscription inside each process
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    cfg = load_config("config.yml")
    global _PARQ_PATH
    _PARQ_PATH = cfg["data"]["exp_return_parquet_path"]

    # --- serial ---
    t0 = time.perf_counter()
    serial_rows = [_run_single(p, cfg) for p in GRID]
    t1 = time.perf_counter()
    df_serial = _stable(pd.DataFrame(serial_rows))

    # --- parallel ---
    n_jobs = min(multiprocessing.cpu_count(), 4)  # modest for test; crank up later
    t2 = time.perf_counter()
    par_rows = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=4, verbose=0)(
        delayed(_run_single)(p, cfg) for p in GRID
    )
    t3 = time.perf_counter()
    df_parallel = _stable(pd.DataFrame(par_rows))

    # --- parity checks ---
    assert len(df_serial) == len(df_parallel) == len(GRID), "Row count mismatch"
    for col in ["n_trades","total_pnl","sharpe"]:
        a, b = df_serial[col].to_numpy(), df_parallel[col].to_numpy()
        if col == "n_trades":
            assert np.array_equal(a, b), f"Mismatch in {col}"
        else:
            assert np.allclose(a, b, rtol=1e-12, atol=1e-12), f"Mismatch in {col}"

    # --- timing + sample print ---
    print(f"\nSERIAL   time: {t1 - t0:.3f}s for {len(GRID)} configs")
    print(f"PARALLEL time: {t3 - t2:.3f}s with n_jobs={n_jobs}")
    print("\nSample results:")
    print(df_serial.to_string(index=False))

    # write a test log (doesn't overwrite your main sweep results)
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out = logs_dir / "gridsearch_results_test_small.csv"
    df_serial.to_csv(out, index=False)
    print(f"\n✅ Wrote {out}")

if __name__ == "__main__":
    main()
