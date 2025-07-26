import itertools
import pandas as pd
import numpy as np
from config_loader import load_config
from src.backtesting.runner_pnl_cv import run_backtest

# Load base config with trained models already included
cfg = load_config("config.yml")

# Logic-only param sweep
thresholds = [0.05, 0.1, 0.2]
stops = [1.0, 1.5, 2.0]
tps = [1.0, 2.0, 3.0]

combinations = list(itertools.product(thresholds, stops, tps))

results = []

print("🔍 Sharpe Ratio Explanation:")
print("  The Sharpe ratio measures return per unit of volatility (risk).")
print("  Sharpe = average(returns) / std(returns) * sqrt(N)")
print("  Higher Sharpe = smoother, more consistent performance.")

for i, (thr, stp, tp) in enumerate(combinations):
    cfg['trading_logic']['threshold_mult'] = thr
    cfg['trading_logic']['entry_threshold'] = thr
    cfg['trading_logic']['stop_loss_atr_mult'] = stp
    cfg['trading_logic']['take_profit_atr_mult'] = tp

    print(f"→ [{i+1}/{len(combinations)}] threshold={thr}, stop={stp}, tp={tp}")

    try:
        metrics, stats = run_backtest(cfg)

        trades = stats.get('trades', {})
        n_trades = (
            trades.get('total', {}).get('total', 0)
            if isinstance(trades, dict)
            else 0
)
        pnl = stats.get('real_pnl', 0)
        sharpe = stats.get('sharpe', None)

        # Early exit filtering
        if n_trades < 3 or pnl <= 0:
            print(f"    ⚠️ Skipping: n_trades={n_trades}, PnL={pnl:.2f}, Sharpe={sharpe}")
            continue

        print(f"    ✅ VALID | Trades: {n_trades}, PnL: {pnl:.2f}, Sharpe: {sharpe:.4f}")

        results.append({
            'threshold_mult': thr,
            'stop_mult': stp,
            'tp_mult': tp,
            'total_pnl': pnl,
            'sharpe': sharpe,
            'n_trades': n_trades
        })

    except Exception as e:
        print(f"    ❌ Error during backtest: {e}")
        continue

# Display results
df = pd.DataFrame(results)

if df.empty:
    print("\\n⚠️ No valid configs passed the filters or all backtests failed.")
else:
    print("\\n=== Valid Configs Sorted by PnL ===")
    print(df.sort_values(by="total_pnl", ascending=False).to_string(index=False))
