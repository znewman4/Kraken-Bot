import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import logging
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from config_loader import load_config
from src.modeling import prepare_features_and_target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_all_models_to_csv(cfg, out_csv='features_with_exp_returns.csv'):
    data_cfg  = cfg['data']
    tl_cfg    = cfg['trading_logic']

    # 1. Load engineered features (as DataFrame)
    csv_fp = Path(data_cfg['feature_data_path'])
    df     = pd.read_csv(csv_fp, parse_dates=['time'])

    # 2. For each horizon, load model and predict using its specific features
    horizon_preds = pd.DataFrame(index=df.index)
    feature_names = {}
    for horizon, model_path in tl_cfg['model_paths'].items():
        logger.info(f"Loading model for horizon {horizon} from {model_path}")
        model = XGBRegressor()
        model.load_model(str(model_path))
        feat_list = model.get_booster().feature_names
        feature_names[horizon] = feat_list

        # Pull only these features, *in this order*
        X = df[feat_list]
        preds = model.predict(X)
        colname = f"exp_return_h{horizon}"
        horizon_preds[colname] = preds

    # 3. Average predictions across all horizons
    horizon_cols = horizon_preds.columns
    df['exp_return'] = horizon_preds.mean(axis=1)      #remember to change the exp_return to a weighted sum
    # Optionally, include all individual horizons
    for col in horizon_cols:
        df[col] = horizon_preds[col]

    # 4. Save
    out_path = Path(out_csv)
    df.to_csv(out_path, index=False)
    logger.info(f"✅ Wrote multi-horizon predictions to: {out_path}")
    return df

if __name__ == "__main__":
    cfg = load_config()
    out_csv = 'data/features_with_exp_returns.csv'
    predict_all_models_to_csv(cfg, out_csv)
