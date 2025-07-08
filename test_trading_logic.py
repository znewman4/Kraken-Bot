#!/usr/bin/env python3
# test_trading_logic1.py
"""
Enhanced trading logic tester:
- Loads engineered OHLCV CSV
- Loads pre-saved horizon models
- Averages their predictions into exp_return
- Computes rolling volatility to normalize edge
- Applies a dynamic, fee-aware threshold for signals
- Sizes positions proportional to confidence
- Calculates total PnL for a given BTC stake
"""
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from src.modeling import prepare_features_and_target


def run_test(
    csv_path='data/processed/btc_ohlcv_5min_engineered.csv',
    model_paths={1: 'models/xgb_h1.json', 5: 'models/xgb_h5.json', 10: 'models/xgb_h10.json'},
    fee_rate=0.002,
    vol_window=20,
    max_position=1.0
):
    # 1) Load engineered data
    df = pd.read_csv(Path(csv_path), parse_dates=['time'], index_col='time')

    # 2) Prepare features (drops any NA target)
    features, _ = prepare_features_and_target(df)

    # 3) Load each horizon model and predict
    preds = pd.DataFrame(index=features.index)
    for h, mfile in model_paths.items():
        mf = Path(mfile)
        if not mf.exists():
            raise FileNotFoundError(f"Model file not found at {mf}")
        m = XGBRegressor()
        m.load_model(str(mf))
        preds[h] = m.predict(features)

    # 4) Compute expected return (simple average)
    exp_return = preds.mean(axis=1).rename('exp_return')

    # 5) Compute rolling volatility of actual closes
    volatility = (
        df['close'].pct_change()
        .rolling(vol_window)
        .std()
        .bfill()
        .rename('volatility')
    )

    # 6) Normalize edge and apply dynamic threshold
    edge_norm = exp_return / volatility
    threshold = (fee_rate / volatility).rename('threshold')

    # 7) Generate discrete signals
    signal = pd.Series(0, index=exp_return.index, name='signal')
    signal[edge_norm >= threshold] = 1
    signal[edge_norm <= -threshold] = -1

    # 8) Size positions proportional to confidence
    conf = edge_norm.abs()
    max_conf = conf.max() if conf.max() > 0 else 1.0
    position = ((conf / max_conf) * max_position * signal).rename('position')

    # 9) Assemble results
    result = pd.DataFrame({
        'close': df['close'],
        'exp_return': exp_return,
        'volatility': volatility,
        'edge_norm': edge_norm,
        'threshold': threshold,
        'signal': signal,
        'position': position
    })

    return result


if __name__ == '__main__':
    # Run the test
    df = run_test()

    # Your BTC stake
    BTC_STAKE = 0.009115

    # 10) Predicted dollar move per BTC
    df['predicted_dollar_move'] = df['exp_return'] * df['close']

    # 11) PnL = shifted position * move * stake
    df['pnl'] = df['position'].shift(1).fillna(0) * df['predicted_dollar_move'] * BTC_STAKE

    # 12) Total P&L
    total_pnl = df['pnl'].sum()

    # Display last few rows and total PnL
    print(
        df[[
            'close','exp_return','volatility','edge_norm','threshold',
            'signal','position','predicted_dollar_move','pnl'
        ]].tail(),
        '\n'
    )
    print(f"💰 Total predicted USD PnL for {BTC_STAKE} BTC: ${total_pnl:.2f}")
