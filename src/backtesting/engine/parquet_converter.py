# src/backtesting/engine/parque_converter.py
# Run this file directly in VS Code (no terminal args). It will:
# 1) load config.yml
# 2) pick a CSV that contains 'exp_return' (pref: exp_return_data_path/backtrader_data_path/feature_data_path)
# 3) save a Parquet file next to the CSV
# 4) print the path you should put in cfg['data']['exp_return_parquet_path']

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

# Make imports robust when you press "Run" from this file
try:
    from config_loader import load_config
except ImportError:
    import sys
    ROOT = Path(__file__).resolve().parents[3]  # repo root: adjust if your tree differs
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from config_loader import load_config  # retry

def _pick_input_csv(cfg: dict) -> Path | None:
    """Choose the best CSV path from config that actually has exp_return."""
    data = cfg.get("data", {})
    # prefer an explicit exp_return CSV if you have one, else backtrader_data_path, else feature_data_path
    candidates = [
        data.get("exp_return_data_path"),
        data.get("backtrader_data_path"),
        data.get("feature_data_path"),
    ]
    for p in candidates:
        if not p:
            continue
        pth = Path(p)
        if pth.suffix.lower() != ".csv" or not pth.exists():
            continue
        # quick sniff for exp_return without reading whole file
        try:
            head = pd.read_csv(pth, nrows=5)
            if "exp_return" in head.columns:
                return pth
        except Exception:
            pass
    return None

def _file_picker_csv() -> Path | None:
    """Fallback: interactive file picker (only if nothing sensible in config)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(
            title="Select CSV with 'exp_return' column",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        return Path(path) if path else None
    except Exception:
        return None

def convert_csv_to_parquet(in_csv: Path, out_parquet: Path | None = None) -> Path:
    df = pd.read_csv(in_csv, parse_dates=["time"]).set_index("time")
    df.index.name = "time"
    if "exp_return" not in df.columns:
        raise ValueError(f"'exp_return' not found in {in_csv}. Columns: {list(df.columns)[:12]} ...")
    if out_parquet is None:
        out_parquet = in_csv.with_suffix(".parquet")
    # write with full precision; pyarrow is preferred
    df.to_parquet(out_parquet, index=True)
    return out_parquet

def main():
    cfg = load_config("config.yml")

    # prefer config-provided CSV that actually contains exp_return
    in_csv = _pick_input_csv(cfg)
    if in_csv is None:
        # optional: let you choose manually (no terminal needed)
        in_csv = _file_picker_csv()
    if in_csv is None:
        raise SystemExit("[ERROR] No suitable CSV found and no file selected.")

    in_csv = Path(in_csv)
    out_parquet = convert_csv_to_parquet(in_csv)

    print("\n[OK] Parquet written ✅")
    print(f"Input CSV : {in_csv}")
    print(f"Output PARQ: {out_parquet}")
    print("\n→ Add this to your config.yml:")
    print("data:")
    print(f"  exp_return_parquet_path: {out_parquet}")
    print("\nTip: Now call the DF-based precomputed runner (no per-run I/O).")

if __name__ == "__main__":
    main()
