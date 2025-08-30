# src/pipelines/rolling_walkforward.py
# Rolling walk-forward retraining + backtest that reuses your existing modules.
# - Mirrors main.py (tune -> train -> SHAP prune -> re-tune/train) per horizon+window
# - Overwrites model files at cfg['trading_logic']['model_paths']
# - Runs runner.py backtest and collects IC / hit / PnL per segment
# - Saves SHAP plots + a summary CSV for easy review
#
# Run:
#   python -m src.pipelines.rolling_walkforward -c config.yml --train-weeks 12 --test-weeks 2 --diag-h 10

from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap,sys
sys.path.append(str(Path(__file__).resolve().parents[2])) 
from config_loader import load_config
from src.modeling import prepare_features_and_target
from src.training import run_tuning, run_training_pipeline
from src.backtesting.runners.runner_slice import run_backtest_df

def weeks_to_bars(weeks: int, interval_minute: int) -> int:
    # Crypto is 24/7; 7*24*60/interval bars per week
    return int(round(weeks * 7 * 24 * (60 / interval_minute)))


def compute_ic_hit(metrics_df: pd.DataFrame, H: int, start, end):
    """Compute H-bar IC & hit over a (start, end] slice, matching your diagnostics logic."""
    m = metrics_df.copy()

    # Canonicalize column names exactly like your diagnostics file
    if 'exp_return' in m.columns and 'exp_r' not in m.columns:
        m = m.rename(columns={'exp_return': 'exp_r'})
    if 'time' in m.columns and 'datetime' not in m.columns:
        m = m.rename(columns={'time': 'datetime'})

    required = ['datetime', 'close', 'exp_r']
    missing = [c for c in required if c not in m.columns]
    if missing:
        raise ValueError(f"[compute_ic_hit] metrics_df missing columns: {missing}. "
                         f"Have: {list(m.columns)}")

    # Parse & index by datetime
    m['datetime'] = pd.to_datetime(m['datetime'], utc=True, errors='coerce').dt.tz_convert(None)
    m = m.sort_values('datetime').set_index('datetime')

    # Slice to test window (start, end]
    start = pd.to_datetime(start, utc=True).tz_convert(None)
    end   = pd.to_datetime(end,   utc=True).tz_convert(None)
    m = m[(m.index > start) & (m.index <= end)].copy()

    # Build realized H-bar return
    m = m.dropna(subset=['exp_r', 'close'])
    m['future_close']     = m['close'].shift(-H)
    m['real_horizon_ret'] = (m['future_close'] - m['close']) / m['close']

    db = m.dropna(subset=['exp_r', 'real_horizon_ret'])
    if db.empty:
        # Helpful debug so you know why you’d get NaNs
        print(f"[compute_ic_hit] Empty slice: "
              f"metrics_range=({metrics_df['datetime'].min()} → {metrics_df['datetime'].max()}), "
              f"window=({start} → {end}), "
              f"len(m_before_dropna)={len(m)}")
        return np.nan, np.nan, 0

    ic  = db['exp_r'].corr(db['real_horizon_ret'])
    hit = (np.sign(db['exp_r']) == np.sign(db['real_horizon_ret'])).mean()
    return float(ic), float(hit), int(len(db))


def ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize time column to a datetime index (handles 'time' or 'datetime')
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
    elif "time" in df.columns:
        dt = pd.to_datetime(df["time"])
    else:
        raise ValueError("Expected a 'time' or 'datetime' column in feature data.")
    df = df.copy()
    df["__dt__"] = dt
    return df.sort_values("__dt__")

def save_shap_summary(shap_values, X, outpath_png: Path, title: str):
    outpath_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath_png, dpi=160)
    plt.close()

def parse_args():
    p = argparse.ArgumentParser("Rolling walk-forward retrain + backtest")
    p.add_argument("-c","--config", default="config.yml")
    p.add_argument("--train-weeks", type=int, default=12)
    p.add_argument("--test-weeks",  type=int, default=2)
    p.add_argument("--diag-h",      type=int, default=10, help="H-bar horizon for diagnostics-style IC/hit")
    p.add_argument("--start-date",  type=str, default=None, help="Optional YYYY-MM-DD slice start")
    p.add_argument("--end-date",    type=str, default=None, help="Optional YYYY-MM-DD slice end")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Paths and intervals
    feature_path = Path(cfg["data"]["feature_data_path"])
    interval_min = int(cfg["exchange"]["interval_minute"])
    model_paths  = cfg["trading_logic"]["model_paths"]   # {"1": "models/xgb_h1.json", ...}
    horizons     = sorted([int(h) for h in model_paths.keys()])

    # Load engineered features once (fast)
    df_all = pd.read_csv(feature_path)
    df_all = ensure_dt(df_all)

    # Optional date slicing to speed up experiments
    if args.start_date:
        df_all = df_all[df_all["__dt__"] >= pd.to_datetime(args.start_date)]
    if args.end_date:
        df_all = df_all[df_all["__dt__"] <  pd.to_datetime(args.end_date)]
    df_all = df_all.reset_index(drop=True)

    # Window sizes in bars
    train_bars = weeks_to_bars(args.train_weeks, interval_min)
    test_bars  = weeks_to_bars(args.test_weeks,  interval_min)

    # Output folder for this run
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("logs") / "rolling" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    n = len(df_all)

    # Slide by test_bars each iteration
    i = train_bars
    k = 0
    while i + test_bars <= n:
        k += 1
        train_start = i - train_bars
        train_end   = i
        test_start  = i
        test_end    = i + test_bars

        df_train = df_all.iloc[train_start:train_end].copy()
        df_test  = df_all.iloc[test_start:test_end].copy()

        w_start = df_train["__dt__"].iloc[0]
        w_mid   = df_train["__dt__"].iloc[-1]
        w_end   = df_test["__dt__"].iloc[-1]

        print(f"\n=== Window {k} | Train: [{w_start} → {w_mid}]  Test: ({w_mid} → {w_end}] ===")

        # --- Retrain per horizon (mirror main.py) ---
        shap_dir = run_dir / f"win_{k:03d}" / "shap"
        for h in horizons:
            # 1) feature/target for THIS horizon on training slice
            cfg["model"]["horizon"] = h
            X_h, y_h = prepare_features_and_target(df_train, cfg["model"])
            X_h = X_h.select_dtypes(include=[np.number]).copy()

            # 2) tuning on full features
            best_params, tuning_path = run_tuning(X_h, y_h, cfg["tuning"], cfg["model"])

            # 3) train + SHAP (full features)
            model_h, shap_vals_h, top_feats_h = run_training_pipeline(
                X_h, y_h, tuning_path,
                cfg["training"], cfg["model"], cfg["selection"]
            )

            # 4) save SHAP summary (full features)
            try:
                save_shap_summary(
                    shap_vals_h, X_h,
                    shap_dir / f"h{h}_full_summary.png",
                    title=f"Window {k} — h={h} (full features)"
                )
            except Exception as e:
                print(f"[warn] SHAP plot (full) failed for h={h}: {e}")

            # 5) re-train on top-K SHAP features, then overwrite model path used by strategy
            X_top_h = X_h[top_feats_h].select_dtypes(include=[np.number]).copy()
            best_params_top, tuning_path_top = run_tuning(
                X_top_h, y_h, cfg["tuning"], cfg["model"]
            )
            model_top_h, shap_vals_top_h, _ = run_training_pipeline(
                X_top_h, y_h, tuning_path_top,
                cfg["training"], cfg["model"], cfg["selection"]
            )

            # optional plot on pruned features
            try:
                save_shap_summary(
                    shap_vals_top_h, X_top_h,
                    shap_dir / f"h{h}_topk_summary.png",
                    title=f"Window {k} — h={h} (top-{len(X_top_h.columns)})"
                )
            except Exception as e:
                print(f"[warn] SHAP plot (top-k) failed for h={h}: {e}")

            # 6) overwrite the exact model file your strategy loads
            out_path = Path(model_paths[str(h)])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(model_top_h, "save_model"):
                model_top_h.save_model(out_path)
            else:
                # fallback: pickle if needed
                import joblib
                joblib.dump(model_top_h, out_path.with_suffix(".pkl"))

            print(f"[saved] h={h} model → {out_path}")

        if cfg.get('calibration', {}).get('enabled', False):
            from src.calibration import fit_calibrators_for_config
            fit_calibrators_for_config(cfg, df_train) 

        # --- Backtest with the freshly overwritten models ---
        df_test_bt = df_test.set_index("__dt__")
        df_test_bt.index.name = "time"
        metrics_df, stats, _ = run_backtest_df(df_test_bt, cfg)

        # --- summarize backtest stats for printing & summary_rows ---
        sharpe   = stats.get("sharpe", np.nan)
        real_pnl = stats.get("real_pnl", np.nan)
        n_trades = stats.get("trades", {}).get("total", {}).get("total", np.nan)
        max_dd   = stats.get("drawdown", {}).get("max", {}).get("drawdown", np.nan)

        H = int(args.diag_h)
        ic_bar, hit_bar, usable = compute_ic_hit(metrics_df, H, start=w_mid, end=w_end)

        print(
            f"Window {k} results → Sharpe: {sharpe:.3f}, PnL: {real_pnl:.2f}, "
            f"Trades: {n_trades}, {H}-bar IC: {ic_bar:.3f}, hit: {0 if np.isnan(hit_bar) else hit_bar:.1%} "
            f"(rows={usable})"
        )
       
        summary_rows.append({
            "window": k,
            "train_start": w_start, "train_end": w_mid, "test_end": w_end,
            "train_bars": train_bars, "test_bars": test_bars,
            "diag_H": H,
            "ic_bar": ic_bar, "hit_bar": hit_bar,
            "sharpe": sharpe, "real_pnl": real_pnl, "n_trades": n_trades, "max_dd": max_dd
        })

        # advance
        i += test_bars

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    out_csv = run_dir / "walkforward_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")
    print(summary_df)

if __name__ == "__main__":
    main()
