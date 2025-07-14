#!/usr/bin/env python
import os
import mlflow
import pandas as pd
from datetime import datetime

from config_loader import load_config
from src.backtesting.runner import run_backtest

# ─── Force MLflow to use your project’s ./mlruns ───────────────────────────
mlruns_path = os.path.abspath("mlruns")
os.makedirs(mlruns_path, exist_ok=True)
mlflow.set_tracking_uri(f"file://{mlruns_path}")
print("MLflow will log to:", mlflow.get_tracking_uri())

def main():
    # 1) Load your config as a dict
    cfg = load_config("config.yml")

    # 2) Set up MLflow experiment & run context
    mlflow.set_experiment(cfg['tracking']['experiment_name'])
    run_name = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=run_name):
        # 3) Log all hyperparameters
        for section in ['trading_logic', 'backtest', 'data', 'model']:
            for k, v in cfg.get(section, {}).items():
                if isinstance(v, dict):
                    for subk, subv in v.items():
                        mlflow.log_param(f"{section}.{k}.{subk}", subv)
                else:
                    mlflow.log_param(f"{section}.{k}", v)

        # 4) Execute the backtest, passing the dict
        metrics_df, _ = run_backtest(cfg)

        # 5) Log summary metrics
        total_pnl = metrics_df['pnl'].sum()
        exp_ret   = metrics_df['exp_return']
        sharpe    = exp_ret.mean() / exp_ret.std() if exp_ret.std() else 0.0
        mlflow.log_metric("total_pnl", total_pnl)
        mlflow.log_metric("sharpe",    sharpe)

        # 6) Save & log the full metrics DataFrame
        metrics_path = "metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        mlflow.log_artifact(metrics_path, artifact_path="metrics")

        # 7) (Optional) Save & log an equity‐curve plot
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(metrics_df.index, metrics_df['pnl'])
            ax.set_title("Equity Curve")
            ax.set_ylabel("PnL")
            fig_path = "equity_curve.png"
            fig.savefig(fig_path)
            mlflow.log_artifact(fig_path, artifact_path="plots")
            plt.close(fig)
        except Exception:
            pass

    print(f"Logged run {run_name} under experiment {cfg['tracking']['experiment_name']}")

if __name__ == "__main__":
    main()
