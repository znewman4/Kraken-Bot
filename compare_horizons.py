# compare_horizons.py

import os
import copy
import yaml
import tempfile

import pandas as pd

from src.backtesting.runner import run_backtest

def compare_horizons(config_path, horizons=[1, 3, 5]):
    # 1) Load the YAML once
    with open(config_path, 'r') as f:
        base_cfg = yaml.safe_load(f)

    # 2) Inspect the original model_paths dict
    orig_paths = base_cfg['trading_logic']['model_paths']
    print("🚦 Original model_paths dict:", orig_paths)
    print("🚦   key types:", [(repr(k), type(k)) for k in orig_paths.keys()])
    print("🚦   dict id:", id(orig_paths))

    rows = []
    for h in horizons:
        print(f"\n🔄 Loop iteration for horizon={h}")

        # 3) Deep-copy the entire config
        cfg = copy.deepcopy(base_cfg)
        curr_mp = cfg['trading_logic']['model_paths']

        # 4) Inspect before override
        print("   before override model_paths:", curr_mp)
        print("   key types:", [(repr(k), type(k)) for k in curr_mp.keys()])
        print("   dict id:", id(curr_mp))

        # 5) Determine the correct key type (int vs str)
        key = h if h in orig_paths else str(h)
        if key not in orig_paths:
            raise KeyError(f"Horizon {h!r} not found in original model_paths keys {list(orig_paths.keys())}")

        # 6) Override model_paths to only this one horizon
        cfg['trading_logic']['model_paths'] = { key: orig_paths[key] }
        after_mp = cfg['trading_logic']['model_paths']

        # 7) Inspect after override
        print("   after override model_paths:", after_mp)
        print("   key types:", [(repr(k), type(k)) for k in after_mp.keys()])
        print("   dict id:", id(after_mp))

        # 8) Dump to a temp YAML
        fd, tmp_path = tempfile.mkstemp(suffix=".yml")
        with os.fdopen(fd, 'w') as tf:
            yaml.safe_dump(cfg, tf)

        print(f"   → backtesting with config file: {tmp_path}")
        metrics, stats, _ = run_backtest(tmp_path)
        os.remove(tmp_path)

        # 9) Extract your key metrics
        total_pnl    = metrics["pnl"].sum()
        sharpe       = stats["sharpe"]
        max_dd       = stats["drawdown"]["max"]["drawdown"]
        n_trades     = stats["trades"]["total"]["total"]
        n_wins       = stats["trades"]["won"]["total"]
        win_rate     = n_wins / n_trades if n_trades else 0.0

        rows.append({
            "horizon":      f"{h}m",
            "total_pnl":    total_pnl,
            "sharpe":       sharpe,
            "max_drawdown": max_dd,
            "n_trades":     n_trades,
            "win_rate":     win_rate,
        })

    # 10) Build and print the comparison table
    df = pd.DataFrame(rows).set_index("horizon")
    print("\n=== Horizon Comparison ===")
    print(df.to_markdown(floatfmt=".3f"))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Compare individual horizon backtests for your KrakenStrategy")
    p.add_argument("--config", "-c", default="config_1m.yml",
                   help="Path to your YAML config file (with all three horizons defined)")
    args = p.parse_args()

    compare_horizons(args.config)
