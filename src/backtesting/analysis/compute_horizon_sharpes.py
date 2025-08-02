#!/usr/bin/env python3
import argparse
import copy
import yaml
import tempfile
import os

from src.backtesting.runners.runner import run_backtest

def main():
    p = argparse.ArgumentParser(
        description="Run per-horizon backtests and compute Sharpe ratios"
    )
    p.add_argument(
        "-c", "--config",
        default="config.yml",
        help="path to master config.yml (default: %(default)s)"
    )
    args = p.parse_args()

    # 1) Load your master config
    with open(args.config) as f:
        cfg_master = yaml.safe_load(f)

    # 2) Extract the list of horizons (as ints)
    horizons = list(map(int, cfg_master["trading_logic"]["model_paths"].keys()))
    results = {}

    # 3) Print header
    print("\n➡️  Running per-horizon backtests...\n")
    print(f"{'Horizon':>7s} │ {'Sharpe':>7s} │ {'PnL':>10s} │ {'Trades':>6s} │ {'Max DD%':>7s}")
    print("-" * 48)

    for h in horizons:
        # 4) Deep-copy master config & override horizon_weights
        cfg = copy.deepcopy(cfg_master)
        cfg["trading_logic"]["horizon_weights"] = [
            1.0 if hh == h else 0.0 for hh in horizons
        ]

        # 5) Loosen gating so we actually get some trades to measure Sharpe:
        cfg["trading_logic"]["threshold_mult"] = 0.0
        cfg["trading_logic"]["persistence"]    = 1

        # 6) Write modified config to temp file
        with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as tmp:
            yaml.safe_dump(cfg, tmp)
            tmp_path = tmp.name

        try:
            # 7) Run backtest by path
            _, stats, _ = run_backtest(tmp_path)
        except Exception as e:
            print(f"  ❌ Horizon {h}: backtest error: {e}")
            os.remove(tmp_path)
            continue
        finally:
            # 8) Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # 9) Safely pull stats (never None)
        sharpe     = stats.get("sharpe",    0.0) or 0.0
        pnl        = stats.get("real_pnl",  0.0) or 0.0
        trades     = stats.get("total_trades", 0) or 0
        max_dd_pct = stats.get("max_dd",    0.0) or 0.0

        results[h] = sharpe

        # 10) Print the line
        print(f"{h:>7d} │ {sharpe:7.3f} │ ${pnl:9.0f} │ {trades:6d} │ {max_dd_pct:7.2f}")

    # 11) Normalize positive Sharpes into suggested weights
    pos_sharpes = {h: max(0.0, s) for h, s in results.items()}
    total_pos   = sum(pos_sharpes.values())
    if total_pos > 0:
        suggested = {h: pos_sharpes[h] / total_pos for h in horizons}
    else:
        # all non-positive → equal weights
        suggested = {h: 1.0 / len(horizons) for h in horizons}

    # 12) Print YAML snippet
    print("\n✅  Suggested `horizon_weights` (normalized positive Sharpe):")
    line = "[" + ", ".join(f"{suggested[h]:.2f}" for h in horizons) + "]"
    print("horizon_weights:", line)
    print("\n👉  Copy that line into your `trading_logic` section of config.yml\n")


if __name__ == "__main__":
    main()
