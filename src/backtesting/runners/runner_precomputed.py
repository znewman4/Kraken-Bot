# src/backtesting/runners/runner_precomputed.py
import os
from contextlib import redirect_stdout, redirect_stderr
from typing import Tuple, Dict

import backtrader as bt
import pandas as pd

from src.backtesting.strategies.strat_precomputed import KrakenStrategy
from src.backtesting.feeds import PrecomputedData
from config_loader import load_config
from src.calibration import ensure_calibrators  # no-ops for precomputed; kept for parity


# ---------- internal helpers (kept parallel to live runner) ----------

def _load_df_from_cfg(cfg: dict) -> pd.DataFrame:
    """Read precomputed data (prefer Parquet), normalize columns, set DatetimeIndex, and apply start/end/max_bars."""
    data = cfg.get("data", {})
    pre_path = data.get("exp_return_parquet_path") or data.get("backtrader_data_path") or data.get("feature_data_path")
    if not pre_path:
        raise ValueError("No data path found in config under data.{exp_return_parquet_path|backtrader_data_path|feature_data_path}.")

    # Read
    if str(pre_path).endswith(".parquet"):
        df = pd.read_parquet(pre_path, engine="pyarrow", memory_map=True)
        if df.index.name != "time" and "time" in df.columns:
            df = df.set_index("time")
    else:
        df = pd.read_csv(pre_path, parse_dates=["time"]).set_index("time")

    df.index.name = "time"

    # Normalize columns for safety (predictor writes them normalized already)
    df.columns = [c.lower().replace(".", "_") for c in df.columns]

    if "exp_return" not in df.columns:
        raise ValueError("Precomputed data must include 'exp_return' (bps).")

    # Slice like live runner
    bt_cfg = cfg.get("backtest", {})
    start = bt_cfg.get("start_date")
    end   = bt_cfg.get("end_date")      # exclusive
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index < pd.to_datetime(end)]

    max_bars = bt_cfg.get("max_bars")
    if max_bars:
        df = df.tail(max_bars)

    return df


def _sanitize_df(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Validate caller-provided DataFrame and apply the same slicing as _load_df_from_cfg."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a pandas.DatetimeIndex")
    if df.index.name != "time":
        df.index.name = "time"
    # normalize cols
    df = df.copy()
    df.columns = [c.lower().replace(".", "_") for c in df.columns]
    if "exp_return" not in df.columns:
        raise ValueError("df must contain 'exp_return' (bps).")

    bt_cfg = cfg.get("backtest", {})
    start = bt_cfg.get("start_date")
    end   = bt_cfg.get("end_date")
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index < pd.to_datetime(end)]

    max_bars = bt_cfg.get("max_bars")
    return df.tail(max_bars) if max_bars else df


def _build_cerebro(cfg: dict, data_feed: bt.feeds.PandasData) -> bt.Cerebro:
    """Construct Cerebro with analyzers/fees/slippage/cash identical to the live runner."""
    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(False)
    cerebro.broker.set_shortcash(True)
    cerebro.adddata(data_feed)

    # analyzers (identical semantics)
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio, _name="sharpe",
        timeframe=bt.TimeFrame.Minutes, compression=5, riskfreerate=0.0
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # === cost wiring identical to live runner ===
    bt_cfg = cfg.get("backtest", {})
    fee  = float(bt_cfg.get("commission", 0.0))
    slip = float(bt_cfg.get("slippage_perc", 0.0))
    cost_bps = (2 * fee + slip) * 1e4

    cerebro.broker.setcommission(commission=fee, leverage=1.0)
    cerebro.broker.set_slippage_perc(
        perc=slip, slip_open=True, slip_limit=True, slip_match=True
    )
    cerebro.broker.setcash(bt_cfg["cash"])

    # Calibrators are no-ops here, but calling keeps parity/visibility
    #ensure_calibrators(cfg)

    # Strategy with same params as live (cost_bps drives EV gate)
    cerebro.addstrategy(KrakenStrategy, config=cfg, cost_bps=cost_bps)
    return cerebro


def _run_quiet(cerebro: bt.Cerebro, cfg: dict):
    """Run Cerebro, optionally silencing stdout/stderr based on cfg['backtest']['quiet']."""
    quiet = cfg.get("backtest", {}).get("quiet", True)
    if quiet:
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            return cerebro.run()
    return cerebro.run()


def _collect_stats(strat: bt.Strategy) -> Dict:
    """Gather the same summary you return today."""
    real_trade_pnls = getattr(strat, "pnls", [])
    sharpe   = (strat.analyzers.sharpe.get_analysis() or {}).get("sharperatio", 0.0) or 0.0
    drawdown = strat.analyzers.drawdown.get_analysis() or {}
    trades   = strat.analyzers.trades.get_analysis() or {}
    return {
        "sharpe": sharpe,
        "drawdown": drawdown,
        "trades": trades,
        "real_pnl": float(sum(real_trade_pnls)) if real_trade_pnls else 0.0,
    }


def _run_core(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, Dict, bt.Cerebro]:
    """Single core used by both public entry points."""
    data = PrecomputedData(dataname=df)
    cerebro = _build_cerebro(cfg, data)
    results = _run_quiet(cerebro, cfg)
    strat = results[0]
    metrics_df = strat.get_metrics()
    stats = _collect_stats(strat)
    return metrics_df, stats, cerebro


# ---------- public API ----------

def run_backtest(config_or_path: str | dict = "config.yml"):
    """Convenience: load data by path from config (prefers Parquet), then run."""
    cfg = config_or_path if isinstance(config_or_path, dict) else load_config(config_or_path)
    df = _load_df_from_cfg(cfg)
    return _run_core(df, cfg)


def run_backtest_df(df: pd.DataFrame, cfg: dict):
    """Fast path for sweeps: caller passes an in-memory DataFrame (no per-run I/O)."""
    dfx = _sanitize_df(df, cfg)
    return _run_core(dfx, cfg)
