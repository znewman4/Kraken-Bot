import pandas as pd
import numpy as np

# --------------- 1. Load Data -------------------
df = pd.read_csv('data/features_with_exp_returns.csv', parse_dates=['time'])
print("Columns in file:", df.columns.tolist())

# --- Calculate ATR if missing ---
if 'atr' not in df.columns:
    # Simple ATR calculation (same as many backtesters): 
    # ATR = rolling mean of True Range over N bars
    N = 14  # Standard ATR window (you can tune)
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(N, min_periods=1).mean()
    print(f"ATR calculated and added with window {N}")


vol_col = 'volatility_5'

# Defensive: check required columns
for col in ['exp_return', 'close', vol_col]:
    if col not in df.columns:
        raise ValueError(f"Column {col} missing from input data!")

H = 29  # forecast horizon (bars)

# --------------- 2. Compute All-Bar Model Performance ---------------
df['future_close'] = df['close'].shift(-H)
df['realized_ret'] = (df['future_close'] - df['close']) / df['close']
allbar_ic = df['exp_return'].corr(df['realized_ret'])
print(f"Raw Model IC (all bars): {allbar_ic:.3f}")

# --------------- 3. Experiment Config ----------------
use_persistence_filter = False
use_quantile_gating   = True
use_volatility_filter = True    # <--- Set to True to turn on!
use_edge_threshold    = False
use_stop_loss         = False  # False = pure H-bar hold
reverse_signal        = False

quantile_window = 100      # rolling history for quantile calculation
entry_quantile = 0.8 
edge_threshold = 0.95
signal_persistence = 2
# Bracket multipliers—try these values, tune as you wish!
stop_mult = 2  # Stop-loss = 1.5x ATR from entry
tp_mult   = 4   # Take-profit = 3.0x ATR from entry

# Defensive: Check for ATR column
if use_stop_loss and 'atr' not in df.columns:
    raise ValueError("ATR column required for stop-loss/take-profit simulation!")


# --- Volatility Filter Setup ---
if use_volatility_filter:
    vol_threshold = df[vol_col].quantile(0.69)
    print(f"70th percentile {vol_col}: {vol_threshold:.6f}")

# --------------- 4. Run Experiment (Non-overlapping Trades) ---------------
results = []
edge_norms = []
sig_buffer = []

i = 0
while i < len(df):
    row = df.iloc[i]
    exp_r = row['exp_return']

    # --- Edge & Signal Calculation ---
    closes = df['close'].iloc[max(0, i-20):i].values
    vol = np.std(np.diff(closes)) if len(closes) >= 3 else np.nan
    edge = exp_r / vol if vol and vol > 0 else 0
    edge_norms.append(edge)
    sig = int(np.sign(edge))
    trade_sig = -sig if reverse_signal else sig

    # --- VOLATILITY FILTER ---
    if use_volatility_filter and row[vol_col] > vol_threshold:
        i += 1
        continue
    if use_edge_threshold and abs(edge) < edge_threshold:
        i += 1
        continue

    # --- Only trade if signal is not zero ---
    if trade_sig == 0:
        i += 1
        continue

    sig_buffer.append(sig)
    if use_persistence_filter and len(sig_buffer) >= signal_persistence:
        # Only proceed if last N signals have all been the same
        if not all(s == sig for s in sig_buffer[-signal_persistence:]):
            i += 1
            continue


    # --- QUANTILE GATING ---
    if use_quantile_gating and i >= quantile_window:
        hist = np.array([e for e in edge_norms[-quantile_window:] if not np.isnan(e)])
        long_thr = np.quantile(hist, entry_quantile)

        short_thr = np.quantile(hist, 1 - entry_quantile)
        if trade_sig == -1 and edge > short_thr:
            i += 1
            continue

        if trade_sig == 1 and edge < long_thr:

            i += 1
            continue

    # --- Simulate Hold to Horizon (Non-overlapping) ---
    if i + H < len(df):
        entry = row['close']
        exit_ = df.iloc[i+H]['close']
        pnl = (exit_ - entry) / entry * trade_sig
        hold_time = H
        stop_hit, tp_hit = False, False  # No stops used
        results.append({
            'entry_time': row['time'],
            'predicted_exp_r': exp_r,
            'net_pnl': pnl,
            'hold_time': hold_time,
            'stop_hit': stop_hit,
            'take_profit_hit': tp_hit,
            'edge': edge,
            'volatility': row[vol_col]
        })
        i += H  # Skip forward H bars for non-overlapping
    else:
        break


# --- Wrap up and Diagnostics ---
tl = pd.DataFrame(results)
print(f"\nTotal trades: {len(tl)}")
if len(tl) > 0:
    print(f"Trade-level IC: {tl['predicted_exp_r'].corr(tl['net_pnl']):.3f}")
    print(f"Avg PnL/trade: {tl['net_pnl'].mean():.4f}")
    print(f"Win rate: {(tl['net_pnl']>0).mean():.1%}")
else:
    print("No trades generated with current experiment settings.")

try:
    import matplotlib.pyplot as plt
    tl['cum_pnl'] = tl['net_pnl'].cumsum()
    tl.set_index('entry_time')['cum_pnl'].plot(title="Cumulative PnL (Non-overlapping, Volatility Filtered)")
    plt.ylabel("Cumulative PnL")
    plt.show()
except ImportError:
    pass
