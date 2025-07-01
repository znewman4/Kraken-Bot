# main.py

from pathlib import Path
import pandas as pd
import time
from datetime import datetime

# Third-party modules
from src.data_loading    import fetch_ohlcv_kraken, save_to_csv, fetch_ohlcv_kraken_paginated
from src.data_cleaning   import clean_ohlcv, validate_ohlcv
from src.technical_engineering import add_technical_indicators, add_return_features
from src.modeling        import prepare_features_and_target
from src.training        import run_tuning, run_training_pipeline



def main():


    # use_cached_data=False
    
    # # Step 1: File path setup
    # data_dir = Path("data/processed")
    # data_dir.mkdir(parents=True, exist_ok=True)
    # file_path = data_dir / "btc_ohlcv_5min_engineered.csv"

  

    # --- SETTINGS ---
    pair     = "XBTUSD"
    interval = 5                 # minutes
    fn       = "btc_ohlcv_5min_engineered.csv"
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / fn

    # --- STEP 1: Load existing data if it exists, get last timestamp ---
    if file_path.exists():
        print("📁 Loading existing engineered data…")
        df_old = pd.read_csv(file_path, index_col=0, parse_dates=True)
        last_ts = df_old.index.max()
        print(f"🔎 Last timestamp in file: {last_ts}")

        # Kraken (via CCXT) wants 'since' in milliseconds
        since_ms = int(last_ts.timestamp() * 1000)

        # --- STEP 2: Fetch only new OHLCV bars ---
        print(f"🌐 Fetching bars since {last_ts} (ms={since_ms})…")
        df_new = fetch_ohlcv_kraken_paginated(pair=pair, interval=interval, since=since_ms)

        if df_new.empty:
            print("ℹ️ No new bars to append.")
            df = df_old
        else:
            # --- STEP 3: Clean & validate new slice ---
            validate_ohlcv(df_new)
            df_new = clean_ohlcv(df_new)
            validate_ohlcv(df_new)

            # --- STEP 4: Combine & dedupe ---
            df = pd.concat([df_old, df_new])
            df = df[~df.index.duplicated(keep="first")].sort_index()
            print(f"✅ Appended {len(df_new)} new rows, total is now {len(df)} rows.")

            # --- STEP 5: Recompute all features (fresh) ---
            df = add_technical_indicators(df)  # RSI, MA, MACD…
            df = add_return_features(df)       # returns, volatility…
            df.dropna(inplace=True)

            # --- STEP 6: Save updated CSV ---
            df.to_csv(file_path)
            print(f"💾 Saved updated data to {file_path}")

    else:
        # No cache yet: do a full fetch + feature engineering
        print("⚙️ No cached file—fetching full history…")
        df = fetch_ohlcv_kraken(pair=pair, interval=interval)
        validate_ohlcv(df)
        df = clean_ohlcv(df)
        validate_ohlcv(df)
        df = add_technical_indicators(df)
        df = add_return_features(df)
        df.dropna(inplace=True)
        df.to_csv(file_path)
        print(f"💾 Initial data saved to {file_path}")

    # --- FINAL PREVIEW ---
    print("✅ Done. Here’s the head of your DataFrame:\n", df.head())


    # if use_cached_data and file_path.exists():
    #     print("📁 Using cached engineered data.")
    #     df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # else:
    #     print("🌐 Fetching fresh data from Kraken...")
    #     df = fetch_ohlcv_kraken(pair='XBTUSD', interval=5)
    #     validate_ohlcv(df)
    #     df = clean_ohlcv(df)
    #     validate_ohlcv(df)
        
    # Step 3: Feature engineering
    df = add_technical_indicators(df)  # RSI, moving averages, MACD, etc.
    df = add_return_features(df)   # log returns, momentum, volatility
    df.dropna(inplace=True)
    save_to_csv(df, file_path)
        
    print(f"✅ Data saved to {file_path}")
    print("✅ Preview:\n", df.head())

    # Step 4: Prepare data
    X, y = prepare_features_and_target(df)

    # Step 5: Hyperparameter tuning
    results, results_path = run_tuning(X, y)  # updated return

    # Step 6: Train final model with best params + explain
    model, model_top, shap_values, top_features = run_training_pipeline(X, y, results_path)
    
    
    # Step 7: Redefine X with top features
    X_top = X[top_features]

    # Step 8: Run tuning again, but on X_top only
    print("\n🔁 Retuning using only top SHAP features...")
    results_top, results_top_path = run_tuning(X_top, y)

    # Step 9: Final model with best params (top features only)
    print("\n🏁 Retraining final model using top features and best params...")
    model_top_final, _, _, _ = run_training_pipeline(X_top, y, results_top_path)
    model.save_model("final_xgb_model.json")
    print("✅ Final model saved as final_xgb_model.json")
    
    
    
    
  


if __name__ == "__main__":
    main()
