# tests/test_precomputed.py
import copy, numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # repo root

from src.backtesting.runners.runner import run_backtest as run_live
from src.backtesting.runners.runner_precomputed import run_backtest as run_pre
from config_loader import load_config

cfg = load_config("config.yml")

# If your live runner expects feature CSVs+models and your precomputed runner expects a CSV with exp_return,
# you don’t need to mutate paths here (each runner reads what it needs).
cfg_live = copy.deepcopy(cfg)
cfg_pre  = copy.deepcopy(cfg)

metrics_live, stats_live, _ = run_live(cfg_live)
metrics_pre,  stats_pre,  _ = run_pre(cfg_pre)

# Robust, apples-to-apples comparison:
# 1) Align on datetime
left  = metrics_live.copy()
right = metrics_pre.copy()

# Make sure the column names match (both strategies already rename to 'exp_return'/'edge_norm' in get_metrics()).
cols_to_compare = [c for c in ('exp_return','edge_norm','signal','position') if c in left.columns and c in right.columns]

merged = pd.merge(left[['datetime']+cols_to_compare],
                  right[['datetime']+cols_to_compare],
                  on='datetime', suffixes=('_live','_pre'))

# Sanity checks
print("Rows (overlap):", len(merged))
for c in cols_to_compare:
    a = merged[f"{c}_live"].to_numpy()
    b = merged[f"{c}_pre"].to_numpy()
    print(f"{c}: allclose ->", np.allclose(a, b, rtol=1e-9, atol=1e-12))

# Optional: compare summary stats
for k in ("real_pnl","sharpe"):
    print(k, stats_live.get(k), stats_pre.get(k))
