#!/usr/bin/env python3
import argparse, copy, yaml, tempfile, os, sys
from pathlib import Path    
sys.path.append(str(Path(__file__).resolve().parents[3]))  
import pandas as pd
import numpy as np
from src.backtesting.runners.runnertest import run_backtest

def extract_flat_stats(stats: dict):
    """Flatten nested Backtrader analyzers into simple keys, safely."""
    sharpe = (stats or {}).get("sharpe", 0.0) or 0.0

    dd     = (stats or {}).get("drawdown", {}) or {}
    maxdd  = (dd.get("max", {}) or {}).get("drawdown", 0.0) or 0.0

    trades = (stats or {}).get("trades", {}) or {}
    ttot   = (trades.get("total", {}) or {}).get("total", 0) or 0

    pnl    = (stats or {}).get("real_pnl", 0.0) or 0.0
    return dict(sharpe=float(sharpe), real_pnl=float(pnl),
                total_trades=int(ttot), max_dd=float(maxdd))

def bar_ic_and_hit(metrics: pd.DataFrame, H: int):
    """
    Compute H-bar ahead IC/Hit using the blended 'exp_return' column
    and the 'close' column (exactly like model_accuracy_diagnostics).
    """
    req = {"close", "exp_return"}
    if not req.issubset(metrics.columns):
        missing = req - set(metrics.columns)
        raise KeyError(f"metrics is missing required columns: {missing}")

    df = metrics.copy()
    df["future_close"] = df["close"].shift(-H)
    df["ret_H"] = (df["future_close"] - df["close"]) / df["close"]
    valid = df.dropna(subset=["exp_return", "ret_H"])
    if len(valid) == 0:
        return np.nan, np.nan, 0
    ic  = valid["exp_return"].corr(valid["ret_H"])
    hit = (np.sign(valid["exp_return"]) == np.sign(valid["ret_H"])).mean()
    return float(ic), float(hit), int(len(valid))

def format_weights(horizons, weights_dict):
    """Return weights in the horizons' order for easy YAML pasting."""
    return "[" + ", ".join(f"{weights_dict[h]:.2f}" for h in horizons) + "]"

def normalize_nonneg(d: dict):
    d2 = {k: max(0.0, float(v)) for k,v in d.items()}
    s  = sum(d2.values())
    if s <= 0:
        # fallback: equal weights
        n = len(d2)
        return {k: 1.0/n for k in d2}
    return {k: d2[k]/s for k in d2}

def run_one_hot(cfg_master: dict, horizons: list[int], h: int):
    """Run a backtest with only horizon h active (weights one-hot)."""
    cfg = copy.deepcopy(cfg_master)
    cfg["trading_logic"]["horizon_weights"] = [1.0 if hh == h else 0.0 for hh in horizons]

    # Optional: loosen gating so we actually get trades (you used this trick too)
    # Adjust if you prefer to keep your live gating:
    cfg["trading_logic"]["threshold_mult"] = cfg["trading_logic"].get("threshold_mult_scan", 0.0)
    cfg["trading_logic"]["persistence"]    = cfg["trading_logic"].get("persistence_scan", 1)

    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp)
        tmp_path = tmp.name

    try:
        metrics, stats, _ = run_backtest(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    flat = extract_flat_stats(stats)
    return metrics, flat

def main():
    ap = argparse.ArgumentParser("Horizon weight sweep: per-h Sharpe + bar-level IC/Hit")
    ap.add_argument("-c", "--config", default="config.yml")
    ap.add_argument("--eval-h", type=int, default=None,
                    help="Evaluation horizon for IC/Hit. "
                         "Default: use each model's own horizon when one-hot.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg_master = yaml.safe_load(f)

    # Establish horizon order (ints) from config model_paths
    horizon_keys = list(cfg_master["trading_logic"]["model_paths"].keys())
    horizons = sorted(map(int, horizon_keys))

    print("\n➡️  Per-horizon backtests (one-hot weights) with Sharpe & IC/Hit\n")
    print(f"{'H':>3s} │ {'Sharpe':>7s} │ {'PnL':>10s} │ {'Trades':>6s} │ {'MaxDD%':>7s} │ {'IC(H)':>6s} │ {'Hit(H)':>7s} │ {'N':>5s}")
    print("-" * 80)

    per_h_sharpe = {}
    per_h_ic     = {}
    per_h_hit    = {}
    rows = []

    for h in horizons:
        metrics, flat = run_one_hot(cfg_master, horizons, h)
        # if eval horizon not provided, evaluate at that model's H
        H_eval = args.eval_h if args.eval_h is not None else h
        ic, hit, n = bar_ic_and_hit(metrics, H_eval)

        per_h_sharpe[h] = flat["sharpe"]
        per_h_ic[h]     = ic
        per_h_hit[h]    = hit

        print(f"{h:>3d} │ {flat['sharpe']:7.3f} │ ${flat['real_pnl']:9.0f} │ "
              f"{flat['total_trades']:6d} │ {flat['max_dd']:7.2f} │ "
              f"{ic:6.3f} │ {hit:7.1%} │ {n:5d}")

        rows.append({
            "H": h,
            "Sharpe": flat["sharpe"],
            "PnL": flat["real_pnl"],
            "Trades": flat["total_trades"],
            "MaxDD%": flat["max_dd"],
            f"IC({H_eval})": ic,
            f"Hit({H_eval})": hit,
            "N": n
        })

    table = pd.DataFrame(rows).sort_values("H")
    # You can save this if you like:
    # table.to_csv("per_horizon_summary.csv", index=False)

    # Build suggested weights
    w_sharpe = normalize_nonneg(per_h_sharpe)
    w_ic     = normalize_nonneg(per_h_ic)

    # Combined suggestion: product (Sharpe × IC), then normalize
    w_combo  = normalize_nonneg({h: per_h_sharpe[h] * max(0.0, per_h_ic[h]) for h in horizons})

    print("\n✅ Suggested `horizon_weights` candidates:")
    print("• Sharpe-normalized:       ", format_weights(horizons, w_sharpe))
    print("• IC-normalized:           ", format_weights(horizons, w_ic))
    print("• Combo (Sharpe × IC):     ", format_weights(horizons, w_combo))
    print("\n👉 Copy one line above into the `trading_logic.horizon_weights` of your config.yml\n")

if __name__ == "__main__":
    main()
