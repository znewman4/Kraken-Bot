import yaml
import os
import copy
import pandas as pd
from config_loader import load_config
from src.backtesting.runner import run_backtest

# 1) Load your base config
base_cfg = load_config("config.yml")

# 2) Define the grid you want to test
threshold_mults = [0.5, 1.0, 1.5, 2.0]
stop_mults      = [1.0, 1.5, 2.0]
tp_mults        = [1.5, 2.0, 2.5]

results = []

# 3) Sweep
for thr_m in threshold_mults:
    for stop_m in stop_mults:
        for tp_m in tp_mults:
            cfg = copy.deepcopy(base_cfg)
            cfg['trading_logic']['threshold_mult']      = thr_m
            cfg['trading_logic']['stop_loss_atr_mult']  = stop_m
            cfg['trading_logic']['take_profit_atr_mult']= tp_m

            # write to temp file
            tmp = "tmp_config.yml"
            with open(tmp, "w") as f:
                yaml.dump(cfg, f)

            # run backtest
            metrics_df, stats, _ = run_backtest(tmp)

            # compute summary from stats
            total_pnl   = stats['real_pnl']
            sharpe      = stats['sharpe']
            n_trades    = stats['trades']['total']['total']
            
            # directional accuracy from metrics_df
            df = metrics_df.copy()
            df['real_ret'] = df['close'].pct_change().shift(-1)
            df = df.dropna(subset=['exp_return', 'real_ret'])
            hits = (df['exp_return'].fillna(0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                    == df['real_ret'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)))
            hit_rate = hits.mean()

            results.append({
                "threshold_mult": thr_m,
                "stop_mult":      stop_m,
                "tp_mult":        tp_m,
                "total_pnl":      total_pnl,
                "sharpe":         sharpe,
                "n_trades":       n_trades,
                "hit_rate":       hit_rate
            })

            os.remove(tmp)

# 4) Display as a DataFrame
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
