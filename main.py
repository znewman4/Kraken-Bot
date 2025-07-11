# main.py

from pathlib import Path
import pandas as pd
import os
import argparse

# 1) Import your new loader
from config_loader import load_config

# 2) Keep your module imports
from src.data_loading            import append_new_ohlcv
from src.data_cleaning           import clean_ohlcv, validate_ohlcv
from src.technical_engineering   import add_technical_indicators, add_return_features
from src.modeling               import prepare_features_and_target
from src.training               import run_tuning, run_training_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run Kraken trading bot")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yml',
        help='Path to your config file (YAML or JSON)'
    )
    parser.add_argument(
        '--set',
        action='append',
        default=[],
        help='Override a config entry, e.g. --set backtest.start_date=2022-01-01'
    )
    return parser.parse_args()


def main():
    # 3) Load CLI args & your central config
    args = parse_args()
    cfg  = load_config(args.config, overrides=args.set)
    print("LOADED CONFIG:", cfg)


    # 4) Pull everything you need from cfg instead of hard-coding:
    pair          = cfg['exchange']['symbol']          # e.g. "XBTUSD"
    interval      = cfg['exchange']['interval_minute'] # e.g. 5
    raw_path      = Path(cfg['data']['raw_data_path'])
    proc_path     = Path(cfg['data']['feature_data_path'])

    # ─── Step 1: load raw OHLCV into `df` ───────────────────────────────────
    df = append_new_ohlcv(pair, interval, raw_path, cfg['exchange'])

    # ─── Step 2: clean & validate ──────────────────────────────────────────
    validate_ohlcv(df)
    df = clean_ohlcv(df, cfg['data'])  # pass data-section if you need any cleaning params
    validate_ohlcv(df)

    # ─── Step 3: engineer features ─────────────────────────────────────────
    df = add_technical_indicators(df, cfg['features'])
    df = add_return_features(df, cfg['features'])
    if cfg['features'].get('drop_na', True):
        df.dropna(inplace=True)

    # ─── Step 4: persist engineered ────────────────────────────────────────
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc_path)
    print(f"💾 Engineered data saved to {proc_path}")

    # ─── Step 5: modeling ───────────────────────────────────────────────────
    print("✅ Data ready—heres the head of `df`:\n", df.head())

    X, y = prepare_features_and_target(df, cfg['model'])
    results, results_path = run_tuning(X, y, cfg['tuning'], cfg['model'])
    model, shap_values, top_features = run_training_pipeline(
    X,
    y,
    results_path,
    cfg['training'],
    cfg['model'],
    cfg['selection']
)

    # ─── Retune & retrain on top SHAP features ──────────────────────────────
    X_top = X[top_features]
    if cfg['selection']['method'] == 'shap':
        print("\n🔁 Retuning using only top SHAP features...")
        results_top, results_top_path = run_tuning(X_top, y, cfg['tuning'], cfg['model'])
        print("\n🏁 Retraining final model using top features and best params...")
        model_top_final, shap_values_top, top_features_top = run_training_pipeline(
            X_top, y, results_top_path,
            cfg['training'],
            cfg['model'],
            cfg['selection']
        )
        os.makedirs(cfg['model']['output_dir'], exist_ok=True)
        model_top_final.save_model(
            os.path.join(cfg['model']['output_dir'], cfg['model']['filename'])
        )
        print(f"✅ Final model saved as {cfg['model']['output_dir']}/{cfg['model']['filename']}")

if __name__ == "__main__":
    main()
