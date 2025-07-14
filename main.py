#main.py

import os
import argparse
import logging
from pathlib import Path

import pandas as pd

from config_loader import load_config
from src.data_loading          import append_new_ohlcv
from src.data_cleaning         import clean_ohlcv, validate_ohlcv
from src.technical_engineering import add_technical_indicators, add_return_features
from src.modeling             import prepare_features_and_target
from src.training             import run_tuning, run_training_pipeline
from src.test_trading_logic   import run_test
from src.backtesting.runner   import run_backtest


def parse_args():
    p = argparse.ArgumentParser("Run Kraken trading bot")
    p.add_argument(
        '--config',
        type=str,
        default='config.yml',
        help='Path to your config file (YAML or JSON)'
    )
    p.add_argument(
        '--set',
        action='append',
        default=[],
        help='Override a config entry, e.g. --set backtest.start_date=2022-01-01'
    )
    p.add_argument(
        '--mode',
        choices=['pipeline', 'backtest'],
        default='pipeline',
        help="pipeline = data+train; backtest = run KrakenStrategy"
    )
    return p.parse_args()


def setup_logging(log_cfg):
    handlers = []
    if log_cfg['handlers']['console']['enabled']:
        ch = logging.StreamHandler()
        ch.setLevel(log_cfg['level'])
        ch.setFormatter(logging.Formatter(log_cfg['format']))
        handlers.append(ch)
    if log_cfg['handlers']['file']['enabled']:
        log_path = Path(log_cfg['handlers']['file']['filename'])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(log_cfg['level'])
        fh.setFormatter(logging.Formatter(log_cfg['format']))
        handlers.append(fh)
    logging.basicConfig(level=log_cfg['level'], handlers=handlers)


def main():
    args = parse_args()
    cfg  = load_config(args.config, overrides=args.set)

    setup_logging(cfg['logging'])
    logger = logging.getLogger(__name__)
    logger.info("Loaded config, starting in %s mode", args.mode)

    if args.mode == 'pipeline':
        # ─── Data Pipeline ────────────────────────────────────────────────
        pair      = cfg['exchange']['symbol']
        interval  = cfg['exchange']['interval_minute']
        raw_path  = Path(cfg['data']['raw_data_path'])
        proc_path = Path(cfg['data']['feature_data_path'])

        df = append_new_ohlcv(pair, interval, raw_path, cfg['exchange'])
        validate_ohlcv(df)
        df = clean_ohlcv(df, cfg['data'])
        validate_ohlcv(df)

        df = add_technical_indicators(df, cfg['features'])
        df = add_return_features(df, cfg['features'])
        if cfg['features'].get('drop_na', True):
            df.dropna(inplace=True)

        proc_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(proc_path)
        logger.info("Engineered data saved to %s", proc_path)

        # ─── Feature Prep & Model Tuning ─────────────────────────────────
        X, y = prepare_features_and_target(df, cfg['model'])
        results, results_path = run_tuning(X, y, cfg['tuning'], cfg['model'])
        model, shap_values, top_features = run_training_pipeline(
            X, y, results_path, cfg['training'], cfg['model'], cfg['selection']
        )

        # ─── SHAP-based Feature Selection & Final Training ────────────────
        if cfg['selection']['method'] == 'shap':
            logger.info("Retuning using only top SHAP features...")
            X_top = X[top_features]

            results_top, results_top_path = run_tuning(
                X_top, y, cfg['tuning'], cfg['model']
            )
            logger.info("Retraining final model using top features and best params...")
            model_top_final, _, _ = run_training_pipeline(
                X_top, y, results_top_path,
                cfg['training'], cfg['model'], cfg['selection']
            )

            os.makedirs(cfg['model']['output_dir'], exist_ok=True)
            final_model_path = os.path.join(
                cfg['model']['output_dir'],
                cfg['model']['filename']
            )
            model_top_final.save_model(final_model_path)
            logger.info("Final model saved as %s", final_model_path)

    else:  # backtest
        # ─── Backtest with KrakenStrategy ─────────────────────────────────
        logger.info("Running backtest using KrakenStrategy…")
        metrics, cerebro = run_backtest(args.config)
        logger.info("Final account value: $%.2f", cerebro.broker.getvalue())
        logger.info("Total PnL:           $%.2f", metrics['pnl'].sum())


if __name__ == "__main__":
    main()
