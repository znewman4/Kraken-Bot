#!/usr/bin/env python3
# model_accuracy_diagnostics.py
# Aligned to your metrics buffer:
# {'datetime','close','exp_r','edge','volatility','signal','threshold'}

import argparse
import os
from pathlib import Path
from datetime import datetime
# --- path bootstrap so 'src' is importable when running by file path ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]  # .../Kraken-Bot
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt

# Avoid seaborn dependencies; pure matplotlib keeps CLI clean.


# ---------- Utils ----------
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ensure_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

def make_outdir(root="logs/diagnostics"):
    tsdir = Path(root) / datetime.now().strftime("%Y%m%d-%H%M%S")
    tsdir.mkdir(parents=True, exist_ok=True)
    return tsdir

def savefig(path):
    path = Path(path)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"[saved] {path}")


# ---------- Main ----------
def main():
    p = argparse.ArgumentParser("H-bar diagnostics (bar & trade level)")
    p.add_argument('-c','--config', default='config.yml', help='Backtest config path')
    p.add_argument('-b','--bars',   type=int, default=10, help='Horizon H in bars')
    p.add_argument('--open', action='store_true', help='Open the output folder when done')
    args = p.parse_args()

    # 1) Run backtest to get metrics (bar-level) + trade_log on disk
    from src.backtesting.runners.runner import run_backtest
    metrics, stats, cerebro = run_backtest(args.config)

    outdir = make_outdir()
    summary_lines = []

    # 2) Bar-level diagnostics from your metrics buffer
    dfm = metrics.copy()

    # Normalize names in case older runs used 'exp_return'/'time'
    if 'exp_return' in dfm.columns and 'exp_r' not in dfm.columns:
        dfm = dfm.rename(columns={'exp_return': 'exp_r'})
    if 'time' in dfm.columns and 'datetime' not in dfm.columns:
        dfm = dfm.rename(columns={'time':'datetime'})

    ensure_cols(dfm, ['datetime','close','exp_r'], "metrics")

    # Index by datetime
    dfm['datetime'] = pd.to_datetime(dfm['datetime'])
    dfm = dfm.sort_values('datetime').set_index('datetime')

    H = args.bars
    dfm['future_close']     = dfm['close'].shift(-H)
    dfm['real_horizon_ret'] = (dfm['future_close'] - dfm['close']) / dfm['close']

    # Basic bar-level skill
    db = dfm.dropna(subset=['exp_r','real_horizon_ret'])
    ic_bar  = db['exp_r'].corr(db['real_horizon_ret'])
    hit_bar = (np.sign(db['exp_r']) == np.sign(db['real_horizon_ret'])).mean()
    print(f"{H}-bar IC (all bars):       {ic_bar:.3f}")
    print(f"{H}-bar hit rate (all bars): {hit_bar:.1%} over {len(db)} bars\n")
    summary_lines += [
        f"{H}-bar IC (bars): {ic_bar:.3f}",
        f"{H}-bar hit (bars): {hit_bar:.1%} over {len(db)} bars",
    ]


    #rolling window plot
    plot_rolling_ic_hit(dfm, outdir, window=50)

    # 3) Load full price series (for trade-level realized H-bar return)
    cfg = load_config(args.config)
    price_path = cfg['data']['feature_data_path']
    prices = pd.read_csv(price_path, parse_dates=['time'])
    prices = prices.set_index('time').sort_index()
    ensure_cols(prices, ['close'], "prices")

    # 4) Read trade log (root or logs/)
    tl_path_candidates = [
        Path('trade_log.csv'),
        Path('logs') / 'trade_log.csv',
        Path('logs') / 'trades' / 'trade_log.csv'
    ]
    tl_path = next((p for p in tl_path_candidates if p.exists()), None)
    if tl_path is None:
        raise FileNotFoundError("Could not find trade_log.csv in project root or logs/.")
    tl = pd.read_csv(tl_path, parse_dates=['entry_time'])
    ensure_cols(tl, ['entry_time','entry_price','position_size','net_pnl','predicted_exp_r'], "trade_log")
    tl = tl.sort_values('entry_time')

    # 4a) Compute H-bar future price/return at entry (direction-aware)
    future_prices = []
    real_rets     = []
    for et, ep, ps in zip(tl['entry_time'], tl['entry_price'], tl['position_size']):
        # Align by exact timestamp; fallback to nearest previous bar if exact not present
        if et in prices.index:
            loc = prices.index.get_loc(et)
        else:
            loc = prices.index.get_indexer([et], method='pad')[0]
            if loc == -1:
                future_prices.append(np.nan)
                real_rets.append(np.nan)
                continue
        idx_future = loc + H
        if idx_future < len(prices):
            fp   = prices['close'].iloc[idx_future]
            sgn  = np.sign(ps)
            real = (fp - ep)/ep * (1 if sgn >= 0 else -1)
        else:
            fp, real = np.nan, np.nan
        future_prices.append(fp)
        real_rets.append(real)

    tl['future_price']     = future_prices
    tl['real_horizon_ret'] = real_rets
    tl = tl.dropna(subset=['real_horizon_ret'])

    # 5) Trade-level diagnostics
    pred = tl['predicted_exp_r']
    real = tl['real_horizon_ret']
    pnl  = tl['net_pnl']
    ic_tr   = pred.corr(real)
    hit_tr  = (np.sign(pred) == np.sign(real)).mean()
    avg_pnl = pnl.mean()
    tot_pnl = pnl.sum()
    win_rt  = (pnl > 0).mean()

    print("Trade-level diagnostics:")
    print(f"  {H}-bar IC on entry: {ic_tr:.3f}")
    print(f"  Hit rate:            {hit_tr:.1%}")
    print(f"  Avg PnL/trade:       {avg_pnl:.2f}")
    print(f"  Total PnL:           {tot_pnl:.2f}")
    print(f"  Win rate:            {win_rt:.1%}\n")
    summary_lines += [
        f"{H}-bar IC (trades): {ic_tr:.3f}",
        f"Hit rate (trades): {hit_tr:.1%}",
        f"Avg PnL/trade: {avg_pnl:.2f}",
        f"Total PnL: {tot_pnl:.2f}",
        f"Win rate: {win_rt:.1%}",
    ]

    # 6) Gate coverage (why no trades in middle?)
    qw = int(cfg['trading_logic'].get('quantile_window', 100))
    q  = float(cfg['trading_logic'].get('entry_quantile', 0.90))

    # Rolling quantiles on bar-level exp_r
    roll = dfm['exp_r'].rolling(qw, min_periods=qw)
    thr_up   = roll.quantile(q)
    thr_down = roll.quantile(1 - q)

    dfm['gate_long']  = dfm['exp_r'] >= thr_up
    dfm['gate_short'] = dfm['exp_r'] <= thr_down
    long_cov  = dfm['gate_long'].mean()
    short_cov = dfm['gate_short'].mean()
    print(f"Gate coverage — long: {long_cov:.1%}, short: {short_cov:.1%}")
    summary_lines += [f"Gate coverage — long: {long_cov:.1%}, short: {short_cov:.1%}"]

    # Did each entry actually clear the gate on that bar?
    # Map entries to dfm index (pad to prior bar if exact not present)
    tl['entry_time_idx'] = [
        et if et in dfm.index else dfm.index[dfm.index.get_indexer([et], method='pad')[0]]
        for et in tl['entry_time']
    ]
    tl['side'] = np.where(tl['position_size'] > 0, 'long', 'short')
    tl['gate_on_entry'] = np.where(
        tl['side'] == 'long',
        dfm.reindex(tl['entry_time_idx'])['gate_long'].to_numpy(),
        dfm.reindex(tl['entry_time_idx'])['gate_short'].to_numpy()
    )
    gate_pass_rate = tl['gate_on_entry'].mean()
    print(f"Entries that actually passed the gate at entry time: {gate_pass_rate:.1%}")
    summary_lines += [f"Gate pass on entries: {gate_pass_rate:.1%}"]

    # 7) Deciles (quintiles) per side for realized return & PnL
    def side_quintiles(df):
        if len(df) < 5:
            return pd.DataFrame()
        df = df.copy()
        df['q'] = pd.qcut(df['predicted_exp_r'], 5, labels=False, duplicates='drop')
        return (df.groupby('q')
                  .agg(mean_real_ret=('real_horizon_ret','mean'),
                       mean_pnl=('net_pnl','mean'),
                       n=('net_pnl','size')))

    long_q = side_quintiles(tl[tl['side']=='long'])
    short_q= side_quintiles(tl[tl['side']=='short'])

    print("\nLong-side quintiles:");  print(long_q.to_string() if not long_q.empty else "(not enough long trades)")
    print("\nShort-side quintiles:"); print(short_q.to_string() if not short_q.empty else "(not enough short trades)")

    # 8) Coverage–lift curve (where should entry_quantile live?)
    qs = np.round(np.linspace(0.70, 0.98, 8), 2)
    rows=[]
    for qq in qs:
        up  = dfm['exp_r'] >= dfm['exp_r'].rolling(qw, min_periods=qw).quantile(qq)
        dn  = dfm['exp_r'] <= dfm['exp_r'].rolling(qw, min_periods=qw).quantile(1-qq)
        mask = (up | dn)
        rr = dfm.loc[mask, 'real_horizon_ret'].dropna()
        rows.append({"q": qq, "coverage": float(mask.mean()), f"mean_ret_{H}bar": float(rr.mean()), "n": int(len(rr))})
    covlift = pd.DataFrame(rows)
    covlift.to_csv(outdir / "coverage_lift.csv", index=False)
    print("\nCoverage–lift table:"); print(covlift.to_string(index=False))

    # ---------- Plots (all saved) ----------
    # PnL histogram
    plt.figure(figsize=(7,4))
    plt.hist(tl['net_pnl'], bins=30, alpha=0.8)
    plt.axvline(0, color='k', ls='--')
    plt.title("Net PnL per Trade"); plt.xlabel("Net PnL"); plt.ylabel("Count")
    savefig(outdir / "hist_pnl_per_trade.png")

    # Predicted exp_r vs Net PnL scatter
    plt.figure(figsize=(7,5))
    plt.scatter(tl['predicted_exp_r'], tl['net_pnl'], alpha=0.7)
    plt.axhline(0, ls='--', c='grey'); plt.axvline(0, ls='--', c='grey')
    plt.title("Predicted exp_r vs Net PnL (trades)")
    plt.xlabel("predicted_exp_r"); plt.ylabel("Net PnL")
    savefig(outdir / "scatter_exp_r_vs_pnl.png")

    # exp_r vs rolling gates with entry markers
    plt.figure(figsize=(11,4))
    plt.plot(dfm.index, dfm['exp_r'], lw=1, alpha=0.8, label="exp_r")
    plt.plot(thr_up.index, thr_up, lw=1, alpha=0.8, label=f"q={q:.2f}")
    plt.plot(thr_down.index, thr_down, lw=1, alpha=0.8, label=f"q={1-q:.2f}")
    # mark entries at exp_r value on that bar
    et_idx = pd.to_datetime(tl['entry_time_idx'])
    exp_at_entry = dfm.reindex(et_idx)['exp_r'].to_numpy()
    plt.scatter(et_idx, exp_at_entry, s=18, label="entries")
    plt.legend(); plt.title("exp_r vs gates (entries overlaid)")
    savefig(outdir / "exp_r_vs_gates_entries.png")

    # Cumulative PnL over time
    tl_sorted = tl.sort_values('entry_time').copy()
    tl_sorted['cum_pnl'] = tl_sorted['net_pnl'].cumsum()
    plt.figure(figsize=(10,4))
    plt.plot(tl_sorted['entry_time'], tl_sorted['cum_pnl'], marker='.')
    plt.title('Cumulative Net PnL Over Time')
    plt.xlabel('Entry Time'); plt.ylabel('Cumulative Net PnL')
    savefig(outdir / "cum_pnl_over_time.png")

    # Stop Distance / ATR distribution (if available)
    if {'stop_dist','atr'}.issubset(tl.columns):
        ratio = (tl['stop_dist'] / tl['atr']).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio):
            plt.figure(figsize=(7,4))
            plt.hist(ratio, bins=20, alpha=0.8)
            plt.title('Stop Distance / ATR'); plt.xlabel('StopDist / ATR'); plt.ylabel('Count')
            savefig(outdir / "hist_stopdist_over_atr.png")

    # Volatility comparison for winners vs losers (box-y)
    if 'volatility' in tl.columns:
        plt.figure(figsize=(7,4))
        box_data = [tl.loc[tl['net_pnl']<=0, 'volatility'].dropna(),
                    tl.loc[tl['net_pnl']>0,  'volatility'].dropna()]
        plt.boxplot(box_data, labels=['Losers','Winners'])
        plt.title('Volatility: Losers vs Winners')
        savefig(outdir / "box_vol_winners_vs_losers.png")

    # 9) Confusion: direction vs profitability + stop/TP attribution
    tl['directional_hit'] = np.sign(tl['predicted_exp_r']) == np.sign(tl['net_pnl'])
    tl['profitable'] = tl['net_pnl'] > 0
    confusion = pd.crosstab(tl['directional_hit'], tl['profitable'])
    print("\nDirection vs Profitability:")
    print(confusion)
    losers = tl[tl['net_pnl'] < 0]
    stopped_out = int(losers['stop_hit'].sum()) if 'stop_hit' in tl.columns else np.nan
    tp_hit      = int(losers['take_profit_hit'].sum()) if 'take_profit_hit' in tl.columns else np.nan
    print(f"\nLosing trades: {len(losers)} | Stopped out: {stopped_out} | TP hit while losing (weird): {tp_hit}")

    # 10) Persist detailed per-trade diagnostics
    out_csv = outdir / "horizon_pnl_diagnostics.csv"
    tl.to_csv(out_csv, index=False)
    print(f"\n✅ Saved trade-level diagnostics → {out_csv}")

    # 11) Save summary text
    (outdir / "coverage_lift.csv").write_text(pd.read_csv(outdir / "coverage_lift.csv").to_csv(index=False))
    with open(outdir / "summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
        f.write("\n\nDirection vs Profitability:\n")
        f.write(confusion.to_string())

    # 12) Optional: auto-open
    if args.open:
        try:
            if os.sys.platform == "darwin":
                os.system(f'open "{outdir}"')
            elif os.sys.platform.startswith("linux"):
                os.system(f'xdg-open "{outdir}"')
            elif os.sys.platform.startswith("win"):
                os.startfile(str(outdir))  # type: ignore
        except Exception as e:
            print(f"(Could not auto-open folder: {e})")

    

def plot_rolling_ic_hit(dfm, outdir, window=50):
        """
        Plot rolling IC and hit rate over a given window size.
        """
        # Only use bars with both prediction and realized return
        db = dfm.dropna(subset=['exp_r','real_horizon_ret']).copy()

        roll_ic = db['exp_r'].rolling(window).corr(db['real_horizon_ret'])
        roll_hit = (
            (np.sign(db['exp_r']) == np.sign(db['real_horizon_ret']))
            .astype(float)
            .rolling(window).mean()
        )

        plt.figure(figsize=(12,6))
        plt.plot(db.index, roll_ic, label=f"Rolling IC ({window} bars)", alpha=0.9)
        plt.plot(db.index, roll_hit, label=f"Rolling Hit Rate ({window} bars)", alpha=0.9)
        plt.axhline(0, color="k", linestyle="--", linewidth=1)
        plt.legend()
        plt.title(f"Rolling IC & Hit Rate (window={window})")
        plt.xlabel("Time")
        plt.ylabel("Value")
        savefig(outdir / "rolling_ic_hit.png")

if __name__ == '__main__':
    main()
