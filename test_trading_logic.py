# src/trading_logic.py

import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from config_loader import load_config
from src.modeling import prepare_features_and_target


def run_test(cfg):
    """
    Run the enhanced trading logic test driven by a central config.
    Expects:
      cfg['data']['feature_data_path']: path to engineered CSV
      cfg['trading_logic'] with keys:
        model_paths: dict[int, str]
        fee_rate: float
        vol_window: int
        max_position: float
        btc_stake: float
    """
    # Slice configs
    data_cfg  = cfg['data']
    tl_cfg    = cfg['trading_logic']
    model_cfg = cfg['model']

    # 1) Load engineered data
    csv_fp = Path(data_cfg['feature_data_path'])
    df     = pd.read_csv(csv_fp, parse_dates=['time'], index_col='time')

    # 2) Prepare features (drops any NA target)
    X, _ = prepare_features_and_target(df, model_cfg)

    # 3) Load each horizon model and predict
    preds = pd.DataFrame(index=X.index)
    for horizon, model_file in tl_cfg['model_paths'].items():
        model = XGBRegressor()
        model.load_model(str(model_file))
        preds[horizon] = model.predict(X)

    # 4) Compute expected return (simple average)
    exp_return = preds.mean(axis=1).rename('exp_return')

    # 5) Compute rolling volatility of closes
    volatility = (
        df['close'].pct_change()
          .rolling(window=tl_cfg['vol_window'])
          .std()
          .bfill()
          .rename('volatility')
    )

    # 6) Normalize edge and dynamic threshold
    edge_norm = (exp_return / volatility).rename('edge_norm')
    threshold = ((tl_cfg['fee_rate'] * 0.5) / volatility).rename('threshold')

    # 7) Generate signals
    signal = pd.Series(0, index=edge_norm.index, name='signal')
    signal[edge_norm >= threshold] = 1
    signal[edge_norm <= -threshold] = -1

    # 8) Size positions proportional to confidence
    conf = edge_norm.abs()
    max_conf = conf.max() if conf.max() > 0 else 1.0
    position = ((conf / max_conf) * tl_cfg['max_position'] * signal).rename('position')

    # 9) Assemble results
    result = pd.concat([
        df['close'], exp_return, volatility,
        edge_norm, threshold, signal, position
    ], axis=1)

    # 10) Predicted dollar move & PnL
    result['predicted_dollar_move'] = result['exp_return'] * result['close']
    stake = tl_cfg.get('btc_stake', 1.0)
    result['pnl'] = result['position'].shift(1).fillna(0) * \
                    result['predicted_dollar_move'] * stake

    return result


if __name__ == '__main__':
    cfg = load_config()
    trade_df = run_test(cfg)
    print(trade_df.tail())
    print(f"💰 Total PnL: ${trade_df['pnl'].sum():.2f}")
