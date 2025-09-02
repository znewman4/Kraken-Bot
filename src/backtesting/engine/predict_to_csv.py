# tools/predict_exp_return.py
import os, sys, logging, json, joblib
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parents[3]))
from config_loader import load_config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("predict_exp_return")


def load_calibrator_if_exists(model_path: str, suffix: str = "_calib.joblib"):
    """Load a calibrator for this model if available, else return None."""
    base, _ = os.path.splitext(model_path)
    cpath = base + suffix
    if os.path.exists(cpath):
        try:
            return joblib.load(cpath)
        except Exception as e:
            log.warning(f"[calib] Failed to load {cpath}: {e}")
    return None


def predict_all_models_to_csv(cfg, out_csv="data/features_with_exp_returns.csv"):
    data_cfg = cfg["data"]
    tl_cfg   = cfg["trading_logic"]

    # Load engineered features
    df = pd.read_csv(Path(data_cfg["feature_data_path"]), parse_dates=["time"]).copy()

    # Normalize columns to match training (strategy does this too)
    df.columns = [c.lower().replace(".", "_") for c in df.columns]

    # Horizon order + weights
    horizons = [int(h) for h in tl_cfg["model_paths"].keys()]
    weights  = np.array(tl_cfg["horizon_weights"], dtype=float)[: len(horizons)]
    wsum     = weights.sum() if weights.sum() != 0 else len(horizons)
    norm_w   = weights / wsum

    horizon_preds = []
    for h in horizons:
        model_path = tl_cfg["model_paths"][str(h)]
        cols_path  = str(model_path) + ".cols.json"

        log.info(f"Loading model h={h} from {model_path}")
        model = XGBRegressor()
        model.load_model(str(model_path))

        # Load the exact trained feature order
        with open(cols_path, "r") as f:
            feats = json.load(f)

        # Slice features in correct order
        X = df[feats].astype(np.float32)

        # Predict in one batch, convert to bps
        preds_bps = model.predict(X) * 1e4

        # Optional calibration
        cal = None
        if cfg.get("calibration", {}).get("enabled", False):
            cal = load_calibrator_if_exists(model_path, cfg["calibration"].get("suffix", "_calib.joblib"))
        if cal is not None:
            try:
                preds_bps = cal.predict(preds_bps.reshape(-1, 1))
            except Exception as e:
                log.warning(f"[calib] WARN predict failed for h={h}: {e}")

        # Save per-horizon prediction
        col = f"exp_return_h{h}"
        df[col] = preds_bps
        horizon_preds.append(col)

    # Weighted average (bps)
    df["exp_return"] = np.dot(df[horizon_preds], norm_w)

    # Save
    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    log.info(f"✅ Wrote predictions to: {out_csv_path} (columns: exp_return + per-horizon)")
    return df


if __name__ == "__main__":
    cfg = load_config("config.yml")
    out_csv = cfg["data"].get("backtrader_data_path", "data/features_with_exp_returns.csv")
    predict_all_models_to_csv(cfg, out_csv)
