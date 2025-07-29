#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import yaml
from src.backtesting.runner import run_backtest

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    p = argparse.ArgumentParser("29-bar horizon & PnL diagnostics")
    p.add_argument('-c','--config', default='config.yml', help='Backtest config')
    p.add_argument('-b','--bars',   type=int, default=29, help='Horizon in bars')
    args = p.parse_args()

    # —————— 1) Run full backtest (with stops, TP, slippage, fees) ——————
    metrics, stats, cerebro = run_backtest(args.config)

    # —————— 2) Bar-level 29-bar diagnostics ——————
    H = args.bars
    dfm = metrics.copy()
    dfm['future_close']     = dfm['close'].shift(-H)
    dfm['real_horizon_ret'] = (dfm['future_close'] - dfm['close'])/dfm['close']
    dfb = dfm.dropna(subset=['exp_return','real_horizon_ret'])
    ic_bar  = dfb['exp_return'].corr(dfb['real_horizon_ret'])
    hit_bar = (np.sign(dfb['exp_return'])==np.sign(dfb['real_horizon_ret'])).mean()
    print(f"{H}-bar IC (all bars):       {ic_bar:.3f}")
    print(f"{H}-bar hit rate (all bars): {hit_bar:.1%} over {len(dfb)} bars\n")

    # —————— 3) Load full price series for trade-level H-bar returns ——————
    cfg = load_config(args.config)
    price_path = cfg['data']['feature_data_path']
    prices = (pd.read_csv(price_path, parse_dates=['time'])
                .set_index('time')
                .sort_index())

    # —————— 4) Read trade_log.csv & compute realized H-bar returns per trade ——————
    tl = pd.read_csv('trade_log.csv', parse_dates=['entry_time'])
    future_prices = []
    real_rets      = []
    for et, ep, ps in zip(tl['entry_time'], tl['entry_price'], tl['position_size']):
        try:
            loc = prices.index.get_loc(et)
            if loc+H < len(prices):
                fp   = prices['close'].iat[loc+H]
                real = (fp - ep)/ep * np.sign(ps)
            else:
                fp, real = np.nan, np.nan
        except KeyError:
            fp, real = np.nan, np.nan
        future_prices.append(fp)
        real_rets.append(real)

    tl['future_price']     = future_prices
    tl['real_horizon_ret'] = real_rets
    tl = tl.dropna(subset=['real_horizon_ret'])

    # —————— 5) Trade-level diagnostics ——————
    pred = tl['predicted_exp_r']
    real = tl['real_horizon_ret']
    pnl  = tl['net_pnl']
    ic_tr   = pred.corr(real)
    hit_tr  = (np.sign(pred)==np.sign(real)).mean()
    avg_pnl = pnl.mean()
    tot_pnl = pnl.sum()
    win_rt  = (pnl>0).mean()

    print("Trade-level diagnostics:")
    print(f"  {H}-bar IC on entry: {ic_tr:.3f}")
    print(f"  Hit rate:         {hit_tr:.1%}")
    print(f"  Avg PnL/trade:    {avg_pnl:.2f}")
    print(f"  Total PnL:        {tot_pnl:.2f}")
    print(f"  Win rate:         {win_rt:.1%}\n")

    # —————— 6) Decile analysis ——————
    tl['decile'] = pd.qcut(pred, 5, labels=False)
    dec = tl.groupby('decile').agg(
        mean_realized_ret = pd.NamedAgg('real_horizon_ret','mean'),
        mean_pnl          = pd.NamedAgg('net_pnl','mean')
    )
    print("Decile summary (realized ret / PnL):")
    print(dec.to_string(),"\n")

    # —————— 7) Save per-trade details ——————
    out = 'horizon_pnl_diagnostics.csv'
    tl.to_csv(out, index=False)
    print(f"✅ Saved detailed diagnostics to {out}")

if __name__=='__main__':
    main()
