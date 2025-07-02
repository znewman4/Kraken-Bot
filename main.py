# main.py

from pathlib import Path
import pandas as pd

from src.data_loading import append_new_ohlcv
from src.data_cleaning      import clean_ohlcv, validate_ohlcv
from src.technical_engineering import add_technical_indicators, add_return_features
from src.modeling           import prepare_features_and_target
from src.training           import run_tuning, run_training_pipeline

def main():


    pair     = "XBTUSD"
    interval = 5  # minutes
    raw_path = Path("data") / "raw" / "btc_ohlcv_5min_raw.csv"
    proc_path = Path("data") / "processed" / "btc_ohlcv_5min_engineered.csv"


    # ─── Step 1: load raw OHLCV into `df` ───────────────────────────────────
    df = append_new_ohlcv(pair, interval, raw_path)

    # ─── Step 2: clean & validate ──────────────────────────────────────────
    validate_ohlcv(df)
    df = clean_ohlcv(df)
    validate_ohlcv(df)

    # ─── Step 3: engineer features ─────────────────────────────────────────
    df = add_technical_indicators(df)
    df = add_return_features(df)
    df.dropna(inplace=True)

    # ─── Step 4: persist engineered ────────────────────────────────────────
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc_path)
    print(f"💾 Engineered data saved to {proc_path}")

    # ─── Step 5: modeling ───────────────────────────────────────────────────
    print("✅ Data ready—here’s the head of `df`:\n", df.head())


    # ─── Final preview & modeling ───────────────────────────────────────────────
    print("✅ Done. Here’s the head of your DataFrame:\n", df.head())

    X, y = prepare_features_and_target(df)
    results, results_path = run_tuning(X, y)
    model, model_top, shap_values, top_features = run_training_pipeline(X, y, results_path)

    # ─── Retune & retrain on top SHAP features ─────────────────────────────────
    X_top = X[top_features]
    print("\n🔁 Retuning using only top SHAP features...")
    results_top, results_top_path = run_tuning(X_top, y)
    print("\n🏁 Retraining final model using top features and best params...")
    model_top_final, _, _, _ = run_training_pipeline(X_top, y, results_top_path)
    model_top_final.save_model("final_xgb_model.json")
    print("✅ Final model saved as final_xgb_model.json")

if __name__ == "__main__":
    main()
