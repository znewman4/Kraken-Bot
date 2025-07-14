import mlflow
from datetime import datetime
import os

from config_loader import load_config
import backtrader as bt
import pandas as pd
from src.backtesting.strategy import KrakenStrategy
from src.backtesting.feeds import EngineeredData

def run_backtest(config):
    """unchanged from before, but now takes a dict not a path"""
    cerebro = bt.Cerebro()
    df = pd.read_csv(
        config['data']['feature_data_path'],
        parse_dates=['time']
    ).set_index('time')
    # optional slicing…
    data = EngineeredData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(KrakenStrategy, config=config)
    cerebro.broker.setcommission(commission=config['trading_logic']['fee_rate'])
    cerebro.broker.set_slippage_perc(
        perc=config['backtest'].get('slippage_perc', 0.0005),
        slip_open=True, slip_limit=True, slip_match=True
    )
    cerebro.broker.setcash(config['trading_logic']['btc_stake'] * df['close'].iloc[0])
    results = cerebro.run()
    strat = results[0]
    return strat.get_metrics(), cerebro

def main(config_path='config.yml'):
    # 1) load config
    config = load_config(config_path)

    # 2) point MLflow at local mlruns/ and select experiment
    mlflow.set_experiment(config['tracking']['experiment_name'])

    # 3) render run_name (uses datetime in your config)
    run_name = eval(f"f'{config['tracking']['run_name']}'")

    # 4) start MLflow run
    with mlflow.start_run(run_name=run_name):
        # 5) log all relevant params
        for section in ['trading_logic', 'backtest', 'data', 'model']:
            if section in config:
                for k, v in config[section].items():
                    # flatten nested dicts if needed
                    if isinstance(v, dict):
                        for subk, subv in v.items():
                            mlflow.log_param(f"{section}.{k}.{subk}", subv)
                    else:
                        mlflow.log_param(f"{section}.{k}", v)

        # 6) run backtest
        metrics_df, cerebro = run_backtest(config)

        # 7) compute summary metrics
        total_pnl = metrics_df['pnl'].sum()
        # simple Sharpe: mean return / std return
        exp_ret = metrics_df['exp_return']
        sharpe = exp_ret.mean() / exp_ret.std() if exp_ret.std() else 0.0
        mlflow.log_metric("total_pnl", total_pnl)
        mlflow.log_metric("sharpe", sharpe)

        # 8) save and log the full metrics DataFrame
        metrics_path = "metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        mlflow.log_artifact(metrics_path, artifact_path="metrics")

        # 9) (optional) save an equity‐curve plot and log it
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

    # 10) done
    print(f"Logged run {run_name} under experiment {config['tracking']['experiment_name']}")

if __name__ == "__main__":
    main()
