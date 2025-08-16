# tests/test_precomputed.py
import copy
import numpy as np
import pandas as pd

from config_loader import load_config
from src.backtesting.runners.runner import run_backtest as run_live
from src.backtesting.runners.runner_precomputed import run_backtest_df as run_pre_df

cfg = load_config("config.yml")

# Live path: keep as-is (reads features/models; computes exp_r on the fly)
cfg_live = copy.deepcopy(cfg)
metrics_live, stats_live, _ = run_live(cfg_live)

# Precomputed path: preload Parquet ONCE, then use DF-based entry
parq = cfg['data']['exp_return_parquet_path']
DF = pd.read_parquet(parq, engine="pyarrow", memory_map=True)
DF.index.name = 'time'

cfg_pre = copy.deepcopy(cfg)
metrics_pre, stats_pre, _ = run_pre_df(DF, cfg_pre)

# Parity checks (same as you did; tight tolerances for decisions)
left, right = metrics_live.copy(), metrics_pre.copy()
cols = [c for c in ('exp_return','edge_norm','signal','position') if c in left.columns and c in right.columns]
merged = pd.merge(left[['datetime']+cols], right[['datetime']+cols], on='datetime', suffixes=('_live','_pre'))
print("Rows (overlap):", len(merged))
for c in cols:
    a, b = merged[f"{c}_live"].to_numpy(), merged[f"{c}_pre"].to_numpy()
    print(f"{c}: allclose ->", np.allclose(a, b, rtol=1e-9, atol=1e-12))
print("real_pnl", stats_live.get("real_pnl"), stats_pre.get("real_pnl"))
print("sharpe",   stats_live.get("sharpe"),   stats_pre.get("sharpe"))
