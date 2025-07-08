#!/usr/bin/env python3
# src/main.py

from pathlib import Path
import pandas as pd
import json
import os

from src.data_cleaning          import clean_ohlcv, validate_ohlcv
from src.technical_engineering  import add_technical_indicators, add_return_features
from src.modeling              import prepare_features_and_target
from src.training              import run_tuning, run_training_pipeline, train_horizon_models

def main():
    pair      = "XBTUSD"
    interval  = 5  # minutes
    raw_path  = Path("data") / "raw" / "btc_ohlcv_5min_raw.csv"
    proc_path = Path("data") / "processed" / "btc_ohlcv_5min_engineered.csv"

    # 1) Load your pre-existing raw CSV (no network fetch)
    df = pd.read_csv(raw_path, parse_dates=['time'], index_col='time')

    # 2) Clean & validate
    validate_ohlcv(df)
    df = clean_ohlcv(df)
    validate_ohlcv(df)

    # 3) Feature engineering
    df = add_technical_indicators(df)
    df = add_return_features(df)
    df.dropna(inplace=True)

    # 4) Persist engineered features
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc_path)
    print(f"💾 Engineered data saved to {proc_path}\n")

    # 5) Modeling: tuning + final training (horizon=1)
    X, y = prepare_features_and_target(df)
    results, results_path = run_tuning(X, y)
    model_full, shap_values, model_top, top_features = run_training_pipeline(X, y, results_path)

    # 6) Retune & retrain on top SHAP features
    X_top = X[top_features]
    print("\n🔁 Retuning using only top SHAP features...")
    results_top, results_top_path = run_tuning(X_top, y)
    print("🏁 Retraining final model using top features...")
    model_top_final, _, _, _ = run_training_pipeline(X_top, y, results_top_path)

    # 7) Save the single-period final model
    os.makedirs("models", exist_ok=True)
    model_top_final.save_model("models/final_xgb_model.json")
    print("✅ Final 1-bar model saved as models/final_xgb_model.json\n")

    # 8) Train and save three horizon models (1, 5, 10) using the full dataset
    best_params = results[0]["params"]
    train_horizon_models(df, best_params, horizons=(1, 5, 10))

if __name__ == "__main__":
    main()
