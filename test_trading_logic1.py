#!/usr/bin/env python3
# test_trading_logic1.py
"""
Simple trading logic tester:
- Loads engineered OHLCV CSV
- Loads pre-saved horizon models properly
- Averages their predictions
- Generates buy/sell signals
"""
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from src.modeling import prepare_features_and_target


def run_test(
    csv_path='data/processed/btc_ohlcv_5min_engineered.csv',
    model_paths={1: 'models/xgb_h1.json', 5: 'models/xgb_h5.json', 10: 'models/xgb_h10.json'},
    buy_threshold=0.001,
    sell_threshold=-0.001
):
    # 1) Load engineered data
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}")
    df = pd.read_csv(path, parse_dates=['time'], index_col='time')

    # 2) Prepare features (drop target)
    features, _ = prepare_features_and_target(df)

    # 3) Load each model and predict correctly
    preds_df = pd.DataFrame(index=features.index)
    for h, model_file in model_paths.items():
        mf = Path(model_file)
        if not mf.exists():
            raise FileNotFoundError(f"Model file not found at {mf}")
        model = XGBRegressor()
        model.load_model(str(mf))      # restore trained model
        preds_df[h] = model.predict(features)

    # 4) Compute average return
    avg_ret = preds_df.mean(axis=1).rename('avg_return')

    # 5) Generate signals
    signals = pd.Series(0, index=avg_ret.index, name='signal')
    signals[avg_ret >= buy_threshold] = 1
    signals[avg_ret <= sell_threshold] = -1

    # 6) Assemble and return
    result = pd.DataFrame({
        'close': df.loc[avg_ret.index, 'close'],
        'avg_return': avg_ret,
        'signal': signals
    })
    return result

if __name__ == '__main__':
    df = run_test()

    # 1) Predicted next‐bar price in USD
    df['predicted_price'] = df['close'] * (1 + df['avg_return'])

    # 2) Predicted dollar move (per BTC)
    df['predicted_dollar_move'] = df['avg_return'] * df['close']

    # 3) Calculate PnL correctly:
    X = 0.009115  # your BTC position size
    # shift signal so you enter on the next bar
    df['shifted_signal'] = df['signal'].shift(1).fillna(0)
    # PnL = position * per-BTC move * signal (from previous bar)
    df['pnl'] = df['shifted_signal'] * df['predicted_dollar_move'] * X

    # 4) Total PnL in USD
    total_pnl = df['pnl'].sum()

    # 5) Display
    print(df[[
        'close',
        'avg_return',
        'predicted_price',
        'predicted_dollar_move',
        'shifted_signal',
        'pnl'
    ]].tail(), '\n')
    print(f"💰 Total predicted PnL: ${total_pnl:.2f}")


