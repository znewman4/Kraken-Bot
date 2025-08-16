# quick_check.py
import numpy as np, pandas as pd, sys
from pathlib import Path    
sys.path.append(str(Path(__file__).resolve().parents[1]))  # push repo root into path
from src.backtesting.runners.runner_precomputed import run_backtest as run_pre
from src.backtesting.runners.runner_precomputed import run_backtest as run_live
from config_loader import load_config

cfg = load_config("config.yml")
# run LIVE (computes XGB inside strategy)
metrics_live, _, _ = run_live(cfg)
# run PRE (reads exp_return)
metrics_pre,  _, _ = run_pre(cfg)

# Compare exp_return in metrics buffers if you log it there,
# otherwise, directly compare the CSV exp_return vs. the strategy’s exp_returns list:
live = np.array(getattr(_, 'exp_returns', []))  # if you expose exp_returns
pre  = pd.read_csv(cfg['data']['backtrader_data_path'])['exp_return'].values[-len(live):]

print("allclose:", np.allclose(live, pre, rtol=1e-10, atol=1e-12))
