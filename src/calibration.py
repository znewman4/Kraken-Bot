# src/calibration.py
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.isotonic import IsotonicRegression

# Uses your existing feature builder
from src.modeling import prepare_features_and_target

def _winsor(y: np.ndarray, mode: str = "quantile", q: float = 0.99, abs_cap: float | None = None) -> np.ndarray:
    y = np.asarray(y, float)
    if mode == "quantile":
        cap = np.quantile(np.abs(y), q)
        return np.clip(y, -cap, cap)
    if mode == "abs" and abs_cap is not None:
        return np.clip(y, -abs_cap, abs_cap)
    return y

class BpsCalibrator:
    """Monotone map: predicted_bps -> realized_bps (per horizon)."""
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_pred_bps, y_true_bps, clip_mode="quantile", clip_q=0.99, abs_cap=None):
        y_pred = np.asarray(y_pred_bps, float)
        y_true = _winsor(np.asarray(y_true_bps, float), mode=clip_mode, q=clip_q, abs_cap=abs_cap)
        self.iso.fit(y_pred, y_true)
        return self

    def predict(self, y_pred_bps):
        return self.iso.predict(np.asarray(y_pred_bps, float))

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.iso, path)

    @staticmethod
    def load(path: str | Path) -> "BpsCalibrator":
        cal = BpsCalibrator()
        cal.iso = joblib.load(path)
        return cal

def _cal_path(model_path: str | Path, suffix: str) -> str:
    p = Path(model_path)
    base = p.with_suffix("")  # drop .json
    return base.as_posix() + suffix

def _needs_fit(model_path: str | Path, cal_path: str | Path) -> bool:
    if not Path(cal_path).exists():
        return True
    return Path(cal_path).stat().st_mtime < Path(model_path).stat().st_mtime  # stale if older than model

def fit_calibrators_for_config(cfg: dict, df: pd.DataFrame | None = None) -> dict[int, str]:
    """Fit & save per-horizon calibrators using last val_frac of the training data."""
    cal_cfg   = cfg.get("calibration", {})
    suffix    = cal_cfg.get("suffix", "_calib.joblib")
    val_frac  = float(cal_cfg.get("val_frac", 0.2))
    clip_mode = cal_cfg.get("clip_mode", "quantile")
    clip_q    = float(cal_cfg.get("clip_q", 0.99))
    min_samples = int(cal_cfg.get("min_samples", 800))

    if df is None:
        # Same dataset you train from
        df = pd.read_csv(cfg["data"]["feature_data_path"])

    saved = {}
    for h_str, model_path in cfg["trading_logic"]["model_paths"].items():
        h = int(h_str)
        # Build features/target exactly like training (already in bps)
        model_cfg = dict(cfg["model"])
        model_cfg["horizon"] = h
        X, y = prepare_features_and_target(df, model_cfg)
        X.columns = [c.lower().replace('.', '_') for c in X.columns]

        # Load XGBoost model and align columns to booster feature order
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        feats = model.get_booster().feature_names
        X = X[feats].select_dtypes(include=[np.number]).copy()

        n = len(X)
        cut = int(n * (1 - val_frac))
        if n - cut < max(1, min_samples):
            #print(f"[calib] skip h={h}: samples={n-cut} < {min_samples}")
            continue

        X_cal, y_cal = X.iloc[cut:], y.iloc[cut:]
        y_pred = model.predict(X_cal)

        cal = BpsCalibrator().fit(y_pred, y_cal, clip_mode=clip_mode, clip_q=clip_q)
        cpath = _cal_path(model_path, suffix)
        cal.save(cpath)
        saved[h] = cpath
        #print(f"[calib] h={h}: saved {cpath} (n={len(X_cal)})")
    return saved

def ensure_calibrators(cfg: dict):
    """No-arg safety: if calibration is enabled and auto_fit is true, fit any missing/stale calibrators.
       Called automatically from runner before starting the backtest."""
    cal_cfg = cfg.get("calibration", {})
    if not cal_cfg.get("enabled", False):
        return  # calibration disabled

    suffix   = cal_cfg.get("suffix", "_calib.joblib")
    auto_fit = cal_cfg.get("auto_fit", True)

    # Check all horizons
    need_any = False
    to_fit = []
    for h_str, model_path in cfg["trading_logic"]["model_paths"].items():
        cpath = _cal_path(model_path, suffix)
        if _needs_fit(model_path, cpath):
            need_any = True
            to_fit.append(int(h_str))

    # if not need_any:
    #     print("[calib] all calibrators present & up-to-date.")
    #     return

    if not auto_fit:
        missing = [h for h in to_fit]
        raise RuntimeError(f"[calib] Missing/stale calibrators for horizons {missing} and auto_fit=False.")

    #print(f"[calib] fitting calibrators for horizons: {to_fit}")
    # Load once to avoid repeated I/O
    df = pd.read_csv(cfg["data"]["feature_data_path"])
    fit_calibrators_for_config(cfg, df=df)
