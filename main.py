# main.py

import os
import argparse
import logging
from pathlib import Path

import pandas as pd
from config_loader import load_config

from src.data_loading import append_new_ohlcv
from src.data_cleaning import clean_ohlcv, validate_ohlcv
from src.technical_engineering import add_technical_indicators, add_return_features
from src.modeling import prepare_features_and_target
from src.training import run_tuning, run_training_pipeline
from src.test_trading_logic import run_test

def parse_args():
    parser = argparse.ArgumentParser(description="Run Kraken trading bot")
    parser.add_argument(
        '--config', type=str, default='config.yml',
        help='Path to your config file (YAML or JSON)'
    )
    parser.add_argument(
        '--set', action='append', default=[],
        help='Override a config entry, e.g. --set backtest.start_date=2022-01-01'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.set)

    # ─── Logging Setup ─────────────────────────────────────────────────────
    log_cfg = cfg['logging']
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
    logger = logging.getLogger(__name__)
    logger.info("Loaded config, starting up")

    # ─── Data Pipeline ──────────────────────────────────────────────────────
    pair      = cfg['exchange']['symbol']
    interval  = cfg['exchange']['interval_minute']
    raw_path  = Path(cfg['data']['raw_data_path'])
    proc_path = Path(cfg['data']['feature_data_path'])

    # ← Pass in backtest.start_date so that if the CSV is missing,
    # it will seed from that date rather than error out.
    df = append_new_ohlcv(
        pair,
        interval,
        raw_path,
        cfg['exchange'],
        cfg['backtest'].get('start_date')
    )

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

    # ─── Modeling Pipeline (one model per horizon) ─────────────────────────
    horizons = cfg['features']['return_horizons']
    for h in horizons:
        logger.info(f"→ Training XGB for horizon={h}")
        # inject horizon into model cfg so prepare_features picks it up
        cfg['model']['horizon'] = h

        # Build X, y
        X, y = prepare_features_and_target(df, cfg['model'])

        # Hyperparameter tuning
        results, results_path = run_tuning(X, y, cfg['tuning'], cfg['model'])
        # Training (and SHAP-based feature selection)
        model, shap_vals, top_feats = run_training_pipeline(
            X, y, results_path, cfg['training'], cfg['model'], cfg['selection']
        )

        if cfg['selection']['method'] == 'shap':
            logger.info("Retuning using only top SHAP features…")
            X_top = X[top_feats]
            results_top, results_top_path = run_tuning(X_top, y, cfg['tuning'], cfg['model'])
            logger.info("Retraining final model using top features…")
            model_top, _, _ = run_training_pipeline(
                X_top, y, results_top_path, cfg['training'], cfg['model'], cfg['selection']
            )
            model = model_top

        # Save horizon-specific model
        out_dir = cfg['model']['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        fname = cfg['model']['filename_template'].format(h=h)
        model_path = os.path.join(out_dir, fname)
        model.save_model(model_path)
        logger.info("Saved model for horizon %d → %s", h, model_path)

if __name__ == "__main__":
    main()
