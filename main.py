# main.py

from pathlib import Path
import pandas as pd
from src.data_loading import fetch_ohlcv_kraken, save_to_csv
from src.data_cleaning import clean_ohlcv, validate_ohlcv
from src.technical_engineering import add_technical_indicators, add_return_features
from src.modeling import prepare_features_and_target
from src.training import run_tuning, run_training_pipeline


def main():
    
    use_cached_data=True
    
    # Step 1: File path setup
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / "btc_ohlcv_1min_engineered.csv"

    if use_cached_data and file_path.exists():
        print("📁 Using cached engineered data.")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    else:
        print("🌐 Fetching fresh data from Kraken...")
        df = fetch_ohlcv_kraken(pair='XBTUSD', interval=1)
        validate_ohlcv(df)
        df = clean_ohlcv(df)
        validate_ohlcv(df)
        
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
        

    
    
    
    
  


if __name__ == "__main__":
    main()
