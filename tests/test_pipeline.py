# tests/test_pipeline.py

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# -----------------------------------------------------------------------------
# Fixture: patch pandas_ta so add_technical_indicators never hits NoneType issues
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_pandas_ta(monkeypatch):
    import pandas_ta as ta

    def ema(series, length):
        return pd.Series(
            np.linspace(series.iloc[0], series.iloc[-1], len(series)),
            index=series.index,
        )

    def sma(series, length):
        return pd.Series(
            np.linspace(series.iloc[0], series.iloc[-1], len(series)),
            index=series.index,
        )

    def rsi(series, length):
        return pd.Series(np.zeros(len(series)), index=series.index)

    def macd(series, fast, slow, signal):
        data = {
            f"MACD_{fast}_{slow}_{signal}": np.zeros(len(series)),
            f"MACDh_{fast}_{slow}_{signal}": np.zeros(len(series)),
            f"MACDs_{fast}_{slow}_{signal}": np.zeros(len(series)),
        }
        return pd.DataFrame(data, index=series.index)

    def bbands(series, length, std):
        # use std as int or float, but key startswith "BBL_20"
        data = {
            f"BBL_{length}_{std}": np.zeros(len(series)),
            f"BBM_{length}_{std}": np.zeros(len(series)),
            f"BBU_{length}_{std}": np.zeros(len(series)),
        }
        return pd.DataFrame(data, index=series.index)

    monkeypatch.setattr(ta, "ema", ema)
    monkeypatch.setattr(ta, "sma", sma)
    monkeypatch.setattr(ta, "rsi", rsi)
    monkeypatch.setattr(ta, "macd", macd)
    monkeypatch.setattr(ta, "bbands", bbands)

    yield

# -----------------------------------------------------------------------------
# Imports under test
# -----------------------------------------------------------------------------
from src.data_cleaning import clean_ohlcv, validate_ohlcv
from src.technical_engineering import add_technical_indicators, add_return_features
from src.modeling import prepare_features_and_target, time_series_cv_split
from src.test_trading_logic import run_test
from src.backtesting.runners.runner import run_backtest


# -----------------------------------------------------------------------------
# 1) Tests for data_cleaning.py
# -----------------------------------------------------------------------------

def make_dirty_df():
    idx = [
        pd.Timestamp("2021-01-03"),
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2021-01-02"),
        pd.Timestamp("2021-01-02"),  # duplicate
    ]
    df = pd.DataFrame({
        "open":   [np.nan, 100,    102,  105],
        "high":   [101,    np.nan, 103,  106],
        "low":    [ 99,    100,    np.nan,104],
        "close":  [100.5,  101.5,  102.5,105.5],
        "volume": [1.0,    2.0,    3.0,   4.0],
    }, index=idx)
    return df

def test_clean_ohlcv_removes_duplicates_and_nans():
    dirty = make_dirty_df()
    cleaned = clean_ohlcv(dirty, data_cfg={})

    assert cleaned.index.is_monotonic_increasing
    assert cleaned.index.duplicated().sum() == 0
    assert not cleaned.isna().any().any()

def test_validate_ohlcv_raises_on_bad_schema():
    good = clean_ohlcv(make_dirty_df(), data_cfg={})
    validate_ohlcv(good)

    bad = good.drop(columns=["open"])
    with pytest.raises(ValueError):
        validate_ohlcv(bad)

    bad2 = good.copy()
    bad2.index = bad2.index.astype(str)
    with pytest.raises(TypeError):
        validate_ohlcv(bad2)

# -----------------------------------------------------------------------------
# 2) Tests for technical_engineering.py
# -----------------------------------------------------------------------------

def make_base_df():
    dates = pd.date_range("2021-01-01", periods=30, freq="D")
    return pd.DataFrame({
        "open":   np.linspace(1, 30, 30),
        "high":   np.linspace(1.1, 30.1, 30),
        "low":    np.linspace(0.9, 29.9, 30),
        "close":  np.linspace(1, 30, 30),
        "volume": np.ones(30),
    }, index=dates)

def test_add_technical_indicators_creates_columns():
    df = make_base_df()
    out = add_technical_indicators(df, features_cfg={})
    # core indicators
    for col in ["ema_10", "sma_10", "rsi_14", "MACD_12_26_9"]:
        assert col in out.columns
    # at least one Bollinger band column starting with "BBL_20"
    assert any(c.startswith("BBL_20") for c in out.columns)

def test_add_return_features_creates_columns():
    df = make_base_df()
    out = add_return_features(df, features_cfg={})
    for col in ["log_return", "log_return_1", "log_return_5", "volatility_5"]:
        assert col in out.columns
    assert isinstance(out["log_return"], pd.Series)

# -----------------------------------------------------------------------------
# 3) Tests for modeling.py
# -----------------------------------------------------------------------------

def make_feature_target_df():
    dates = pd.date_range("2021-01-01", periods=5, freq="D")
    return pd.DataFrame({
        "close": np.arange(5, dtype=float),
        "feat1": np.arange(5, dtype=float) * 2
    }, index=dates)

def test_prepare_features_and_target_shape():
    df = make_feature_target_df()
    X, y = prepare_features_and_target(df, model_cfg={})
    assert len(X) == len(df) - 1
    assert len(y) == len(df) - 1
    assert "feat1" in X.columns

def test_time_series_cv_split_counts_and_order():
    df = make_feature_target_df()
    X, y = prepare_features_and_target(df, model_cfg={})
    splits = list(time_series_cv_split(X, y, n_splits=2, test_size=0.5))
    assert len(splits) == 2
    for train_idx, test_idx in splits:
        assert len(test_idx) > 0
        if len(train_idx) > 0:
            assert max(train_idx) < min(test_idx)

# -----------------------------------------------------------------------------
# 4) End-to-end test for trading logic
# -----------------------------------------------------------------------------

class DummyXGB:
    def load_model(self, path): pass
    def predict(self, X): return np.zeros(len(X))

@pytest.fixture(autouse=True)
def patch_xgb(monkeypatch):
    import src.test_trading_logic as tl
    monkeypatch.setattr(tl, "XGBRegressor", DummyXGB)
    yield

def test_run_test_end_to_end(tmp_path):
    dates = pd.date_range("2021-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "time":   dates,
        "open":   np.arange(5, dtype=float),
        "high":   np.arange(5, dtype=float) + 0.5,
        "low":    np.arange(5, dtype=float) - 0.5,
        "close":  np.arange(5, dtype=float),
        "volume": np.ones(5),
    })
    csv_fp = tmp_path / "data.csv"
    df.to_csv(csv_fp, index=False)

    cfg = {
        "data": {"feature_data_path": str(csv_fp)},
        "trading_logic": {
            "model_paths": {1: "dummy"},
            "fee_rate":    0.01,
            "vol_window":  2,
            "max_position": 1.0,
            "btc_stake":    1.0,
        },
        "model": {"horizon": 1},
    }

    result = run_test(cfg)
    expected = {
        "close", "exp_return", "volatility", "edge_norm",
        "threshold", "signal", "position",
        "predicted_dollar_move", "pnl"
    }
    assert expected.issubset(result.columns)
    assert pd.api.types.is_numeric_dtype(result["pnl"])

#Tests for backtesting pipeline


def test_equity_survives():
    metrics, cerebro = run_backtest("config.yml")
    assert cerebro.broker.getvalue() > 0
    assert metrics['pnl'].sum() > -1000  # adjust for expectations

def test_no_excessive_drawdown():
    metrics, cerebro = run_backtest("config.yml")
    equity_curve = metrics['pnl'].cumsum()
    drawdown = equity_curve.cummax() - equity_curve
    assert drawdown.max() < 0.5 * cerebro.broker.getvalue()

