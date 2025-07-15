import os
import argparse
import logging
import subprocess

import wandb
from pathlib import Path
import pandas as pd

from config_loader             import load_config
from src.data_loading          import append_new_ohlcv
from src.data_cleaning         import clean_ohlcv, validate_ohlcv
from src.technical_engineering import add_technical_indicators, add_return_features
from src.modeling              import prepare_features_and_target
from src.training              import run_tuning, run_training_pipeline
from src.backtesting.runner    import run_backtest
# if your run_kraken_strategy script exposes a Python entrypoint, import it
# from run_kraken_strategy      import main as run_kraken_strategy22

def parse_args():
    p = argparse.ArgumentParser("Run Kraken trading bot")
    p.add_argument('--config', type=str, default='config.yml')
    p.add_argument('--set',    action='append', default=[])
    p.add_argument('--mode',   choices=['pipeline','backtest'], default='pipeline')
    return p.parse_args()

def setup_logging(log_cfg):
    handlers = []
    if log_cfg['handlers']['console']['enabled']:
        ch = logging.StreamHandler()
        ch.setLevel(log_cfg['level'])
        ch.setFormatter(logging.Formatter(log_cfg['format']))
        handlers.append(ch)
    if log_cfg['handlers']['file']['enabled']:
        fp = Path(log_cfg['handlers']['file']['filename'])
        fp.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(fp)
        fh.setLevel(log_cfg['level'])
        fh.setFormatter(logging.Formatter(log_cfg['format']))
        handlers.append(fh)
    logging.basicConfig(level=log_cfg['level'], handlers=handlers)

def main():
    args = parse_args()
    cfg  = load_config(args.config, overrides=args.set)

    # 1) Standard logging
    setup_logging(cfg['logging'])
    logger = logging.getLogger(__name__)
    logger.info("Loaded config, starting in %s mode", args.mode)

    # 2) Initialize W&B
    wandb.init(
        project=cfg['tracking']['wandb']['project'],
        entity=cfg['tracking']['wandb'].get('entity'),
        config=cfg
    )

    # 3) Run pipeline or backtest
    if args.mode == 'pipeline':
        # ─── Data + Train ───────────────────────────────────────
        df = append_new_ohlcv(
            cfg['exchange']['symbol'],
            cfg['exchange']['interval_minute'],
            Path(cfg['data']['raw_data_path']),
            cfg['exchange']
        )
        validate_ohlcv(df)
        df = clean_ohlcv(df, cfg['data'])
        validate_ohlcv(df)

        df = add_technical_indicators(df, cfg['features'])
        df = add_return_features(df, cfg['features'])
        if cfg['features'].get('drop_na', True):
            df.dropna(inplace=True)

        # save features
        proc_path = Path(cfg['data']['feature_data_path'])
        proc_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(proc_path, index=False)
        logger.info("Engineered data saved to %s", proc_path)

        X, y = prepare_features_and_target(df, cfg['model'])
        _, tune_path = run_tuning(X, y, cfg['tuning'], cfg['model'])
        model, shap_vals, top_feats = run_training_pipeline(
            X, y, tune_path,
            cfg['training'], cfg['model'], cfg['selection']
        )

        # Compute & log metrics
        equity, trades = run_backtest(cfg)[:2]  # adapt if your API differs
        metrics = {
            "total_pnl":       equity.iloc[-1] - equity.iloc[0],
            "sharpe":          compute_sharpe(equity),
            "max_drawdown":    compute_max_drawdown(equity),
            "win_rate":        compute_win_rate(trades),
            "num_trades":      len(trades),
        }
        # PnL distribution stats
        stats = trades["pnl"].describe().to_dict()
        metrics.update({f"pnl_{k}": float(v) for k,v in stats.items()})
        wandb.log(metrics)

        # Artifacts
        eq_csv = "equity_curve.csv"
        equity.to_frame("equity").to_csv(eq_csv, index=False)
        wandb.save(eq_csv)

        fi = pd.Series(model.feature_importances_, index=cfg['features']['names'])
        fi_csv = "feature_importances.csv"
        fi.to_frame("importance").to_csv(fi_csv)
        wandb.save(fi_csv)

        # Save model
        model_path = os.path.join(cfg['model']['output_dir'], cfg['model']['filename'])
        os.makedirs(cfg['model']['output_dir'], exist_ok=True)
        model.save_model(model_path)
        wandb.save(model_path)

    else:
        # ─── Backtest Only ───────────────────────────────────────
        metrics, cerebro = run_backtest(cfg)
        summary = {
            "total_pnl": metrics['pnl'].sum(),
            "final_value": cerebro.broker.getvalue()
        }
        wandb.log(summary)
        logger.info("Backtest complete: %s", summary)

    # 4) Finally, call your existing run_kraken_strategy script (if you want)
    # Option A: as a Python import
    # strategy_metrics = run_kraken_strategy()
    # wandb.log(strategy_metrics)

    # Option B: as a subprocess (will pick up any prints/logs)
    subprocess.run(["python", "run_kraken_strategy.py"], check=True)

    # 5) Finish W&B
    wandb.finish()


if __name__ == "__main__":
    main()
