#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
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

    # Directional hit: was model prediction sign same as realized trade return (net PnL)?
    tl['directional_hit'] = np.sign(tl['predicted_exp_r']) == np.sign(tl['net_pnl'])
    tl['profitable'] = tl['net_pnl'] > 0

    # Confusion matrix: Direction vs. Profitability
    confusion = pd.crosstab(tl['directional_hit'], tl['profitable'])
    print("Direction vs. Profitability confusion matrix:")
    print(confusion, "\n")

    # Attribution for losses: Stop loss / Take profit
    losers = tl[tl['net_pnl'] < 0]
    stopped_out = losers['stop_hit'].sum()
    tp_hit = losers['take_profit_hit'].sum()
    print(f"Of {len(losers)} losing trades:")
    print(f"  Stopped out:      {stopped_out}")
    print(f"  Take profit hit:  {tp_hit}")
    print(f"  Others:           {len(losers) - stopped_out - tp_hit}\n")

    # Outlier analysis
    print("5 Biggest Losing Trades:")
    print(tl.nsmallest(5, 'net_pnl')[['entry_time','predicted_exp_r','net_pnl','stop_hit','take_profit_hit','hold_time_mins']])
    print("\n5 Biggest Winning Trades:")
    print(tl.nlargest(5, 'net_pnl')[['entry_time','predicted_exp_r','net_pnl','stop_hit','take_profit_hit','hold_time_mins']])

    # PnL histogram
    plt.figure(figsize=(7,4))
    plt.hist(tl['net_pnl'], bins=30, alpha=0.7)
    plt.axvline(0, color='k', linestyle='--')
    plt.title("Histogram of Net PnL per Trade")
    plt.xlabel("Net PnL")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Predicted vs Net PnL scatter
    plt.figure(figsize=(7,5))
    plt.scatter(tl['predicted_exp_r'], tl['net_pnl'], alpha=0.7)
    plt.axhline(0, ls='--', c='grey')
    plt.axvline(0, ls='--', c='grey')
    plt.title("Predicted exp_r vs Net PnL (per trade)")
    plt.xlabel("Predicted exp_r")
    plt.ylabel("Net PnL")
    plt.tight_layout()
    plt.show()

    # 1. Same Bar Exits
    same_bar_exits = (tl['hold_time_mins'] == 0).mean()
    print(f"\nTrades exited same bar as entry: {same_bar_exits:.1%}")

    # 2. Histogram of Stop Distance as % of ATR
    plt.figure(figsize=(7,4))
    plt.hist(tl['stop_dist'] / tl['atr'], bins=20, color='royalblue', alpha=0.7)
    plt.title('Histogram: Stop Distance as % of ATR')
    plt.xlabel('Stop Distance / ATR')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # 3. Average volatility for stop-out vs. profitable trades
    stopped = tl[tl['stop_hit'] == True]
    profitable = tl[tl['net_pnl'] > 0]
    mean_vol_stopped = stopped['volatility'].mean()
    mean_vol_prof = profitable['volatility'].mean()
    print(f"Average volatility on stop-outs:   {mean_vol_stopped:.6f}")
    print(f"Average volatility on winners:     {mean_vol_prof:.6f}")

    # 4. Distribution of predicted returns for trades taken
    plt.figure(figsize=(7,4))
    sns.histplot(tl['predicted_exp_r'], bins=20, kde=True)
    plt.title('Predicted exp_r for All Trades')
    plt.xlabel('Predicted exp_r')
    plt.tight_layout()
    plt.show()

    # 5. PnL Attribution: Slippage and Commissions
    # If you have slippage or commission columns, uncomment below and adjust:
    # print(f"Mean slippage: {tl['slippage'].mean():.2f}")
    # print(f"Mean commission: {tl['commission'].mean():.2f}")

    # 6. Cumulative PnL over time
    tl_sorted = tl.sort_values('entry_time')
    tl_sorted['cum_pnl'] = tl_sorted['net_pnl'].cumsum()
    plt.figure(figsize=(10,5))
    plt.plot(tl_sorted['entry_time'], tl_sorted['cum_pnl'], marker='.')
    plt.title('Cumulative Net PnL Over Time')
    plt.xlabel('Entry Time')
    plt.ylabel('Cumulative Net PnL')
    plt.tight_layout()
    plt.show()

    # 7. Stop Distance vs ATR for stop-outs vs. profitable trades
    plt.figure(figsize=(7,4))
    sns.boxplot(data=tl, x='profitable', y=tl['stop_dist']/tl['atr'])
    plt.title('Stop Distance/ATR: Profitable vs Stopped-Out Trades')
    plt.xlabel('Profitable Trade')
    plt.ylabel('Stop Distance / ATR')
    plt.tight_layout()
    plt.show()

    # 8. Volatility for stop-outs vs. winners (boxplot)
    plt.figure(figsize=(7,4))
    sns.boxplot(data=tl, x='profitable', y='volatility')
    plt.title('Volatility: Profitable vs Stopped-Out Trades')
    plt.xlabel('Profitable Trade')
    plt.ylabel('Volatility')
    plt.tight_layout()
    plt.show()

    # 9. Outlier trade summary for easy inspection
    print("\nOutlier Trade Analysis:")
    print("5 Biggest Losing Trades:")
    print(tl.nsmallest(5, 'net_pnl')[['entry_time','predicted_exp_r','net_pnl','stop_hit','take_profit_hit','hold_time_mins']])
    print("\n5 Biggest Winning Trades:")
    print(tl.nlargest(5, 'net_pnl')[['entry_time','predicted_exp_r','net_pnl','stop_hit','take_profit_hit','hold_time_mins']])

    # 10. Reminder: If you have additional columns for slippage/comm, print those stats here!

    # 11. Recap: Proportion of all trades that are stop-outs, take-profits, or neither
    print(f"\nProportion stopped out:    {tl['stop_hit'].mean():.1%}")
    print(f"Proportion take profit:    {tl['take_profit_hit'].mean():.1%}")
    print(f"Proportion neither:        {1 - tl['stop_hit'].mean() - tl['take_profit_hit'].mean():.1%}")

    # 12. Optional: Plot of holding times
    plt.figure(figsize=(7,4))
    plt.hist(tl['hold_time_mins'], bins=20, color='salmon', alpha=0.8)
    plt.title("Histogram: Hold Time (mins)")
    plt.xlabel("Hold Time (mins)")
    plt.tight_layout()
    plt.show()

    # Example: Only analyze trades where volatility is in the bottom 50%
    low_vol_threshold = tl['volatility'].quantile(0.5)
    low_vol_trades = tl[tl['volatility'] < low_vol_threshold]
    print(f"Trades in low volatility regime: {len(low_vol_trades)} / {len(tl)}")

    # Rerun win rate and avg PnL for this subset
    print("Low-vol regime diagnostics:")
    print(f"  Win rate: {(low_vol_trades['net_pnl'] > 0).mean():.1%}")
    print(f"  Avg PnL/trade: {low_vol_trades['net_pnl'].mean():.2f}")



if __name__=='__main__':
    main()
