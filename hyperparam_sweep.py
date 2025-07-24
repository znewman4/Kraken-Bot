#!/usr/bin/env python
import yaml, os, copy
import pandas as pd
from config_loader import load_config
from src.backtesting.runner import run_backtest

# 1) Load your base config
base_cfg = load_config("config.yml")

# 2) Define the grid you want to test
stop_mults = [1.0, 1.5, 2.0]
tp_mults   = [1.5, 2.0, 2.5]

results = []

# 3) Sweep
for stop_m in stop_mults:
    for tp_m in tp_mults:
        cfg = copy.deepcopy(base_cfg)  # shallow copy
        cfg['trading_logic']['stop_loss_atr_mult']    = stop_m
        cfg['trading_logic']['take_profit_atr_mult'] = tp_m

        # write to temp file
        tmp = "tmp_config.yml"
        with open(tmp, "w") as f:
            yaml.dump(cfg, f)

        # run backtest
        metrics, _ = run_backtest(tmp)

        # compute summary
        pnl   = metrics['pnl'].sum()
        sharpe = metrics['pnl'].mean() / (metrics['pnl'].std() + 1e-8) * (252**0.5)
        trades = (metrics['signal'] != 0).sum()

        results.append({
            "stop_mult": stop_m,
            "tp_mult":   tp_m,
            "total_pnl": pnl,
            "sharpe":    sharpe,
            "n_trades":  trades
        })

        os.remove(tmp)

# 4) Display as a DataFrame
df = pd.DataFrame(results)
print(df.to_string(index=False))
