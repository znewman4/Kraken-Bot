# tools/predict_exp_return.py
import os, sys, logging
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
sys.path.append(str(Path(__file__).resolve().parents[3]))

from config_loader import load_config
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("predict_exp_return")

def predict_all_models_to_csv(cfg, out_csv='data/features_with_exp_returns.csv'):
    data_cfg = cfg['data']
    tl_cfg   = cfg['trading_logic']

    df = pd.read_csv(Path(data_cfg['feature_data_path']), parse_dates=['time'])
    df = df.copy()  # avoid SettingWithCopy shenanigans

    # Establish a deterministic horizon order that matches your config + weights
    horizons = [int(h) for h in tl_cfg['model_paths'].keys()]  # preserves YAML order
    weights  = np.array(cfg['trading_logic']['horizon_weights'], dtype=float)[:len(horizons)]
    wsum     = weights.sum() if weights.sum() != 0 else len(horizons)
    norm_w   = weights / wsum

    # Predict per horizon, using each model's feature list (and order)
    horizon_preds = []
    for h in horizons:
        model_path = tl_cfg['model_paths'][str(h)]
        log.info(f"Loading model h={h} from {model_path}")
        model = XGBRegressor()
        model.load_model(str(model_path))
        feats = model.get_booster().feature_names
        # if any feature is missing, this should raise now (same behavior you’d get in-strategy)
        X = df[feats]
        preds = model.predict(X)
        col = f"exp_return_h{h}"
        df[col] = preds
        horizon_preds.append(col)

    # Weighted average (exactly what the strategy does)
    # exp_return = sum(pred_h[i] * w[i]) / sum(w)
    df['exp_return'] = np.sum([df[col] * w for col, w in zip(horizon_preds, norm_w)], axis=0)

    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    log.info(f"✅ Wrote predictions to: {out_csv_path} (columns: exp_return + per-horizon)")
    return df

if __name__ == "__main__":
    cfg = load_config("config.yml")
    predict_all_models_to_csv(cfg, cfg['data'].get('backtrader_data_path', 'data/features_with_exp_returns.csv'))
