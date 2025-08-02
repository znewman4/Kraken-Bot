import pandas as pd
from src.modeling import prepare_features_and_target
from config_loader import load_config
from src.backtesting.tuners.custom_tuning import run_custom_tuning

# STEP 1: Load engineered OHLCV data
df = pd.read_csv("data/processed/btc_ohlcv_5min_engineered.csv", parse_dates=["time"], index_col="time")

# STEP 2: Load config
config = load_config("config.yml")

# STEP 3: Generate features and targets
X, y = prepare_features_and_target(df, config["model"])

# STEP 4: Run PnL-based hyperparameter + logic tuning
run_custom_tuning(X, y, config)
