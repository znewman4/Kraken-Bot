import json
import tempfile
import pandas as pd
from pathlib import Path
import tempfile
import pandas as pd
from pathlib import Path
from src.backtesting.runner_pnl_cv import run_backtest

class PnLBacktestScorer:
    def __init__(self, base_config, result_key="real_pnl"):
        self.base_config = base_config
        self.result_key = result_key

    def __call__(self, model, X_val, y_val, logic_overrides):
        # Save the model to a temporary path
        model_path = Path(tempfile.mktemp(suffix=".json"))
        model.save_model(model_path)

        # Load full dataset so we can reattach OHLCV + engineered features
        full_df = pd.read_csv(self.base_config['data']['feature_data_path'], parse_dates=['time'], index_col='time')

        # Select only the validation window
        df_val_full = full_df.loc[X_val.index].copy()

        # Attach the model's regression target (required for backtest sim logic)
        df_val_full['target'] = y_val

        # Save this to temp CSV
        data_path = Path(tempfile.mktemp(suffix=".csv"))
        df_val_full.to_csv(data_path)

        # Patch config for this fold
        cfg = json.loads(json.dumps(self.base_config))  # deep copy
        cfg["model"]["filename"] = str(model_path)
        cfg["data"]["feature_data_path"] = str(data_path)
        cfg["trading_logic"].update(logic_overrides)

        # Run the fast backtest
        metrics, stats = run_backtest(cfg)
        return stats.get(self.result_key, -999)




