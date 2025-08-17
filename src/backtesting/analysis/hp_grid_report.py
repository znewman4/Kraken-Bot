# src/backtesting/analysis/hp_grid_report.py
"""
Hyperparameter grid search report:
- Loads logs/gridsearch_results.csv (or --csv path)
- Prints concise analysis & suggestions
- Saves plots into logs/hp_report/<timestamp>/

Run:
  python -m src.backtesting.analysis.hp_grid_report
  # or
  python src/backtesting/analysis/hp_grid_report.py --csv logs/gridsearch_results.csv
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------ I/O ------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=None,
                   help="Path to gridsearch_results.csv (default: logs/gridsearch_results.csv)")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory for figures (default: logs/hp_report/<timestamp>/)")
    p.add_argument("--top_frac", type=float, default=0.15,
                   help="Top fraction used to propose next grid (default 0.15)")
    p.add_argument("--score_w_sharpe", type=float, default=1000.0,
                   help="Composite score = sharpe*W + total_pnl (default 1000)")
    return p.parse_args()


def default_paths(args_csv: str | None, args_outdir: str | None):
    repo_root = Path(__file__).resolve().parents[3]
    csv_path = Path(args_csv) if args_csv else (repo_root / "logs" / "gridsearch_results.csv")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args_outdir) if args_outdir else (repo_root / "logs" / "hp_report" / ts)
    outdir.mkdir(parents=True, exist_ok=True)
    return csv_path, outdir


def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find results CSV at: {csv_path}")
    df = pd.read_csv(csv_path)
    # normalize column names (if someone used slightly different casing)
    df.columns = [c.strip() for c in df.columns]
    # enforce dtypes for safety
    num_cols = ["quantile_window","entry_quantile","min_trade_size","threshold_mult",
                "max_hold_bars","stop_loss_mult","take_profit_mult","n_trades",
                "total_pnl","sharpe"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # composite score to rank (tweakable)
    return df


# ------------------------ Analysis helpers ------------------------

PARAMS = ("quantile_window","entry_quantile","min_trade_size","threshold_mult",
          "max_hold_bars","stop_loss_mult","take_profit_mult")

def add_composite_score(df: pd.DataFrame, w_sharpe: float = 1000.0) -> pd.DataFrame:
    df = df.copy()
    df["score"] = df["sharpe"].fillna(-1e9) * float(w_sharpe) + df["total_pnl"].fillna(-1e9)
    return df

def stable_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(list(PARAMS)).reset_index(drop=True)

def describe_param_space(df: pd.DataFrame) -> pd.DataFrame:
    info = []
    for p in PARAMS:
        if p in df.columns:
            vals = df[p].dropna().unique()
            info.append((p, len(vals), np.min(vals), np.max(vals), np.sort(vals)[:10]))
    return pd.DataFrame(info, columns=["param","n_unique","min","max","example_values"])

def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated (maximize sharpe & total_pnl)."""
    pts = df[["sharpe","total_pnl"]].to_numpy()
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not keep[i]:
            continue
        # dominated if exists j with both >= and at least one >
        dominated = (pts[:,0] >= pts[i,0]) & (pts[:,1] >= pts[i,1]) & (
                    (pts[:,0] > pts[i,0]) | (pts[:,1] > pts[i,1]))
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return df.loc[keep]

def propose_next_grid(df: pd.DataFrame, top_frac: float = 0.15, min_levels=2, max_levels=6) -> dict:
    """Pick parameter values that occur in the top X% by composite score."""
    n = max(1, int(len(df) * top_frac))
    top = df.nlargest(n, "score")
    proposal = {}
    for p in PARAMS:
        if p in top.columns:
            uniq = sorted(set(top[p].dropna().tolist()))
            if len(uniq) > max_levels:
                # keep middle-ish spread (quantiles)
                qs = np.linspace(0.0, 1.0, max_levels)
                uniq = sorted(set(np.quantile(top[p], qs)))
            if len(uniq) < min_levels and p in df.columns:
                # pad by adding closest neighbors from full space
                full = sorted(set(df[p].dropna().tolist()))
                for v in full:
                    if v not in uniq:
                        uniq.append(v)
                    if len(uniq) >= min_levels:
                        break
            proposal[p] = sorted(uniq)
    return proposal


# ------------------------ Plotting ------------------------

def _savefig(outdir: Path, name: str):
    path = outdir / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path

def plot_scatter_front(df: pd.DataFrame, outdir: Path):
    """Sharpe vs PnL, bubble size ~ n_trades, mark Pareto."""
    fig = plt.figure(figsize=(7,5))
    ax = plt.gca()
    x = df["total_pnl"].to_numpy()
    y = df["sharpe"].to_numpy()
    s = 10 + 2*np.sqrt(np.maximum(df["n_trades"], 0.0).to_numpy() + 1e-9)
    ax.scatter(x, y, s=s, alpha=0.6)
    ax.set_xlabel("Total PnL")
    ax.set_ylabel("Sharpe")
    ax.set_title("Outcome scatter & Pareto frontier")
    # Pareto highlight
    pf = pareto_front(df)
    ax.scatter(pf["total_pnl"], pf["sharpe"], s=40, marker='x')
    return _savefig(outdir, "01_scatter_pareto")

def plot_box_by_param(df: pd.DataFrame, metric: str, outdir: Path):
    """Boxplots of metric distribution per level for each param."""
    for p in PARAMS:
        if p not in df.columns: 
            continue
        fig = plt.figure(figsize=(7,4))
        ax = plt.gca()
        # keep order sorted
        data = []
        labels = []
        for val in sorted(df[p].dropna().unique()):
            data.append(df.loc[df[p]==val, metric].dropna().to_numpy())
            labels.append(str(val))
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_title(f"{metric} by {p}")
        ax.set_xlabel(p)
        ax.set_ylabel(metric)
        _savefig(outdir, f"02_box_{metric}_by_{p}")

def plot_pair_heatmap(df: pd.DataFrame, a: str, b: str, metric: str, outdir: Path):
    """Heatmap of mean(metric) over (a,b)."""
    if a not in df.columns or b not in df.columns: 
        return
    pivot = pd.pivot_table(df, index=a, columns=b, values=metric, aggfunc="mean")
    fig = plt.figure(figsize=(6,5))
    ax = plt.gca()
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel(b)
    ax.set_ylabel(a)
    ax.set_title(f"Mean {metric}: {a} × {b}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _savefig(outdir, f"03_heat_{metric}_{a}_x_{b}")

def plot_interaction_panels(df: pd.DataFrame, driver: str, facet: str, metric: str, outdir: Path):
    """Lines of metric vs driver, faceted by 'facet' values."""
    if driver not in df.columns or facet not in df.columns:
        return
    # create one figure per facet value
    for fval in sorted(df[facet].dropna().unique()):
        fig = plt.figure(figsize=(6,4))
        ax = plt.gca()
        sub = df[df[facet]==fval]
        means = sub.groupby(driver, as_index=False)[metric].mean().sort_values(driver)
        ax.plot(means[driver], means[metric], marker="o")
        ax.set_title(f"{metric} vs {driver} | {facet}={fval}")
        ax.set_xlabel(driver)
        ax.set_ylabel(metric)
        _savefig(outdir, f"04_line_{metric}_vs_{driver}_facet_{facet}_{fval}")


# ------------------------ Main report ------------------------

def main():
    args = parse_args()
    csv_path, outdir = default_paths(args.csv, args.outdir)
    df = load_results(csv_path)
    df = add_composite_score(df, w_sharpe=args.score_w_sharpe)

    # Basic health
    print("\n=== Grid Summary ===")
    print(f"Rows: {len(df)}   Columns: {list(df.columns)}")
    print(describe_param_space(df).to_string(index=False))

    # Leaders
    print("\n=== Top by Sharpe ===")
    print(df.sort_values("sharpe", ascending=False).head(10).to_string(index=False))
    print("\n=== Top by Total PnL ===")
    print(df.sort_values("total_pnl", ascending=False).head(10).to_string(index=False))
    print("\n=== Top by Composite (Sharpe*W + PnL) ===")
    print(df.sort_values("score", ascending=False).head(10).to_string(index=False))

    # Sensitivity (1-way)
    print("\n=== One-way Sensitivities: mean(metric) per level ===")
    for metric in ("sharpe","total_pnl"):
        for p in PARAMS:
            if p not in df.columns: continue
            means = df.groupby(p, as_index=False)[metric].mean().sort_values(metric, ascending=False)
            print(f"\n[ {metric} ] by {p}")
            print(means.to_string(index=False))

    # Pairwise interactions (2-way)
    print("\n=== Strong Pairwise Interactions (top cells) ===")
    pairs = [
        ("threshold_mult", "max_hold_bars"),
        ("threshold_mult", "take_profit_mult"),
        ("stop_loss_mult", "take_profit_mult"),
        ("quantile_window", "threshold_mult"),
        ("entry_quantile", "threshold_mult"),
    ]
    for a,b in pairs:
        if a in df.columns and b in df.columns:
            pv = pd.pivot_table(df, index=a, columns=b, values="score", aggfunc="mean")
            # report top 3 cells
            flat = pv.stack().sort_values(ascending=False)
            print(f"\nPair: {a} × {b}  [top 3 by score]")
            print(flat.head(3))

    # Propose next grid (from best fraction)
    proposal = propose_next_grid(df, top_frac=args.top_frac)
    print("\n=== Proposed next grid (from top fraction) ===")
    for k, v in proposal.items():
        print(f"{k}: {v}")

    # Plots
    print(f"\nWriting figures to: {outdir}")
    plot_scatter_front(df, outdir)
    for metric in ("sharpe", "total_pnl"):
        plot_box_by_param(df, metric, outdir)


    # Heatmaps that matter most for trading decisions
    for metric in ("sharpe", "total_pnl"):
        plot_pair_heatmap(df, "threshold_mult", "max_hold_bars", metric, outdir)
        plot_pair_heatmap(df, "stop_loss_mult", "take_profit_mult", metric, outdir)
        plot_pair_heatmap(df, "quantile_window", "threshold_mult", metric, outdir)

    # Interaction lines: metric vs driver, faceted by another param
    for metric in ("sharpe", "total_pnl"):
        plot_interaction_panels(df, driver="threshold_mult", facet="max_hold_bars", metric=metric, outdir=outdir)
        plot_interaction_panels(df, driver="take_profit_mult", facet="stop_loss_mult", metric=metric, outdir=outdir)

    print("\n✅ Report complete.")



if __name__ == "__main__":
    main()
