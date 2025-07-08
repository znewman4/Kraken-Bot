# src/test_trading_logic.py
"""
Standalone trading logic tester:
Loads processed OHLCV data, applies multi-horizon regression models, combines predictions,
normalizes by rolling volatility, applies fee-aware thresholds, and sizes positions.
Everything returns in a new results DataFrame; your original CSVs and models stay unchanged.
"""
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

# Import project modules with src prefix
from src.modeling import prepare_features_and_target


def load_data(csv_path):
    """
    Load processed OHLCV data directly from CSV.

    Args:
      csv_path: path to processed OHLCV CSV file (str or Path)
    Returns:
      DataFrame with OHLCV data indexed by time
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed CSV not found at {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['time'], index_col='time')
    return df


def load_models(model_paths):
    """
    Load one XGBRegressor per horizon.

    Args:
      model_paths: dict mapping horizon (int) to model file path (str or Path)
    Returns:
      dict mapping horizon to loaded XGBRegressor
    """
    models = {}
    for horizon, path in model_paths.items():
        model_file = Path(path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found at {model_file}")
        m = XGBRegressor()
        m.load_model(str(model_file))
        models[horizon] = m
    return models


def predict_multi(models, features):
    """
    Predict returns for each horizon.

    Args:
      models: dict of horizon->XGBRegressor
      features: DataFrame of input features
    Returns:
      DataFrame of predictions with one column per horizon
    """
    preds = {h: m.predict(features) for h, m in models.items()}
    return pd.DataFrame(preds, index=features.index)


def combine_predictions(preds, weights):
    """
    Compute weighted average of horizon predictions.

    Args:
      preds: DataFrame from predict_multi
      weights: dict mapping horizon to weight (float)
    Returns:
      Series of expected returns
    """
    total_weight = sum(weights.values())
    exp_vals = sum(preds[h] * (weights[h] / total_weight) for h in weights)
    return pd.Series(exp_vals, index=preds.index, name='exp_return')


def compute_volatility(data, window):
    """
    Calculate rolling volatility of close returns.

    Args:
      data: DataFrame with a 'close' column
      window: look-back period for volatility (int)
    Returns:
      Series of volatility
    """
    returns = data['close'].pct_change()
    volatility = returns.rolling(window).std().bfill()
    return pd.Series(volatility, index=data.index, name='volatility')


def generate_signals(exp_return, volatility, fee_rate):
    """
    Create buy/sell signals based on normalized edge and fees.

    Args:
      exp_return: Series of expected returns
      volatility: Series of volatility values
      fee_rate: flat fee fraction per trade (float)
    Returns:
      Series of signals (-1, 0, 1)
    """
    edge = exp_return / volatility
    threshold = fee_rate / volatility
    signals = pd.Series(0, index=edge.index, name='signal')
    signals[edge >= threshold] = 1
    signals[edge <= -threshold] = -1
    return signals


def size_positions(exp_return, volatility, signals, max_position):
    """
    Size positions proportional to signal confidence.

    Args:
      exp_return: Series of expected returns
      volatility: Series of volatility values
      signals: Series of signals (-1,0,1)
      max_position: maximum position size (float)
    Returns:
      Series of position sizes (float)
    """
    confidence = (exp_return / volatility).abs()
    max_conf = confidence.max() if confidence.max() > 0 else 1.0
    sizes = (confidence / max_conf) * max_position
    return pd.Series(sizes * signals, index=signals.index, name='position')


def run_trading_logic(
    csv_path,
    model_paths,
    weights,
    fee_rate=0.002,
    vol_window=20,
    max_position=1.0
):
    """
    Full pipeline:
      1) load_data
      2) prepare_features_and_target
      3) load_models
      4) predict_multi
      5) combine_predictions
      6) compute_volatility
      7) generate_signals
      8) size_positions

    Args:
      csv_path: path to processed CSV file
      model_paths: dict horizon->model file path
      weights: dict horizon->weight for combining predictions
      fee_rate: flat fee fraction per trade
      vol_window: window size for volatility calculation
      max_position: maximum position size
    Returns:
      DataFrame with 'close','exp_return','volatility','signal','position'
    """
    data = load_data(csv_path)
    features, _ = prepare_features_and_target(data)
    data = data.loc[features.index]
    models = load_models(model_paths)
    preds = predict_multi(models, features)
    exp_return = combine_predictions(preds, weights)
    volatility = compute_volatility(data, vol_window)
    signals = generate_signals(exp_return, volatility, fee_rate)
    positions = size_positions(exp_return, volatility, signals, max_position)

    result = pd.DataFrame({
        'close': data['close'],
        'exp_return': exp_return,
        'volatility': volatility,
        'signal': signals,
        'position': positions
    }, index=features.index)
    return result


if __name__ == '__main__':
    # Example configuration
    processed_csv = Path('data/processed/btc_ohlcv_5min_engineered.csv')
    model_paths = {1: 'models/xgb_h1.json', 5: 'models/xgb_h5.json', 10: 'models/xgb_h10.json'}
    weights = {1: 0.5, 5: 0.3, 10: 0.2}
    df = run_trading_logic(
        csv_path=processed_csv,
        model_paths=model_paths,
        weights=weights
    )
    print(df.tail())
