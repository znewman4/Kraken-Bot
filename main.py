# src/main.py
import os
import argparse
import logging
from pathlib import Path

import pandas as pd

from config_loader import load_config
from src import bulk_download
from src.data_cleaning import clean_ohlcv, validate_ohlcv
from src.technical_engineering import add_technical_indicators, add_return_features
from src.modeling import prepare_features_and_target
from src.training import run_tuning, run_training_pipeline


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
    interval = cfg['exchange']['interval_minute']
    raw_path = Path(cfg['data']['raw_data_path'])
    proc_path = Path(cfg['data']['feature_data_path'])

    bulk_download.main()

    # 1) Load raw CSV
    df = pd.read_csv(raw_path, parse_dates=["time"])
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)

    # 2) Validate raw download
    validate_ohlcv(df, minutes=interval, label="raw_after_download")

    # 3) Clean and re-validate
    df = clean_ohlcv(df, cfg['data'])
    validate_ohlcv(df, minutes=interval, label="cleaned")

    # 4) Feature engineering
    df = add_technical_indicators(df, cfg['features'])
    df = add_return_features(df, cfg['features'])
    if cfg['features'].get('drop_na', True):
        df.dropna(inplace=True)

    # 5) Validate after feature engineering
    validate_ohlcv(df, minutes=interval, label="after_features")

    # Save engineered dataset
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc_path)
    logger.info("Engineered data saved to %s", proc_path)



    # ─── Data Slice ────────────────────────────────────────────────────
    tr_cfg = cfg.get('training', {})
    start = tr_cfg.get('start_date')  # e.g., "2025-06-01"
    end   = tr_cfg.get('end_date')    # exclusive

    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index < pd.to_datetime(end)]

    # ─── Horizon Ensemble Training & Saving ─────────────────────────────────
    for h_str, model_path in cfg['trading_logic']['model_paths'].items():
        h = int(h_str)
        cfg['model']['horizon'] = h

        # 1) build features & target for this horizon
        X_h, y_h = prepare_features_and_target(df, cfg['model'])

        # 2) hyperparameter tuning
        best_params, tuning_path = run_tuning(X_h, y_h, cfg['tuning'], cfg['model'])

        # 3) train final model on full set
        model_h, shap_vals_h, top_feats_h = run_training_pipeline(
            X_h, y_h, tuning_path,
            cfg['training'], cfg['model'], cfg['selection']
        )

        # 4) save the model to the path strategy expects
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model_h.save_model(model_path)
        logger.info(f"Horizon {h} model saved to {model_path}")

        # 5) Optional SHAP-based Retraining per horizon
        if cfg['selection']['method'] == 'shap':
            logger.info("Retraining horizon %s using top SHAP features...", h)
            X_top_h = X_h[top_feats_h]
            best_params_top, tuning_path_top = run_tuning(
                X_top_h, y_h, cfg['tuning'], cfg['model']
            )
            model_top_h, _, _ = run_training_pipeline(
                X_top_h, y_h, tuning_path_top,
                cfg['training'], cfg['model'], cfg['selection']
            )
            # overwrite the same file so strategy picks up SHAP model
            model_top_h.save_model(model_path)
            logger.info("Horizon %s SHAP-retrained model saved to %s", h, model_path)

    if cfg.get('calibration', {}).get('enabled', False):
        from src.calibration import fit_calibrators_for_config
        fit_calibrators_for_config(cfg, df) 

if __name__ == "__main__":
    main()
