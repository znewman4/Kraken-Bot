#runner_precomputed.py
import os
from contextlib import redirect_stdout, redirect_stderr
from typing import Tuple, Dict

import backtrader as bt
import pandas as pd

from src.backtesting.strategies.strat_precomputed import KrakenStrategy
from src.backtesting.feeds import PrecomputedData
from config_loader import load_config


# ---------- internal helpers (single source of truth) ----------

def _load_df_from_cfg(cfg: dict) -> pd.DataFrame:
    """Read precomputed data (prefer Parquet), ensure Backtrader-friendly index, validate columns, trim to max_bars."""
    pre_path = (
        cfg["data"].get("exp_return_parquet_path")
        or cfg["data"].get("backtrader_data_path")
        or cfg["data"]["feature_data_path"]
    )

    if str(pre_path).endswith(".parquet"):
        df = pd.read_parquet(pre_path, engine="pyarrow", memory_map=True)
        # If the parquet kept 'time' as a column, set it as index
        if df.index.name != "time":
            if "time" in df.columns:
                df = df.set_index("time")
            df.index.name = "time"
    else:
        df = pd.read_csv(pre_path, parse_dates=["time"]).set_index("time")
        df.index.name = "time"

    if "exp_return" not in df.columns:
        raise ValueError(
            "Precomputed runner needs 'exp_return' in the data. "
            f"Got columns: {list(df.columns)[:12]}..."
        )

    max_bars = cfg.get("backtest", {}).get("max_bars")
    if max_bars:
        df = df.tail(max_bars)
    return df


def _sanitize_df(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Validate a caller-provided DataFrame and apply max_bars. Used by run_backtest_df."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a pandas.DatetimeIndex")
    if df.index.name != "time":
        df.index.name = "time"
    if "exp_return" not in df.columns:
        raise ValueError("df must contain 'exp_return'")
    max_bars = cfg.get("backtest", {}).get("max_bars")
    return df.tail(max_bars) if max_bars else df


def _build_cerebro(cfg: dict, data_feed: bt.feeds.PandasData) -> bt.Cerebro:
    """Construct Cerebro with analyzers/fees/slippage/cash identical to the live runner."""
    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(False)
    cerebro.broker.set_shortcash(True)
    cerebro.adddata(data_feed)

    # analyzers (keep identical semantics)
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio, _name="sharpe",
        timeframe=bt.TimeFrame.Minutes, compression=5, riskfreerate=0.0
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    cerebro.addstrategy(KrakenStrategy, config=cfg)

    cerebro.broker.setcommission(commission=cfg["trading_logic"]["fee_rate"], leverage=1.0)
    cerebro.broker.set_slippage_perc(
        perc=cfg["backtest"].get("slippage_perc", 0.0005),
        slip_open=True, slip_limit=True, slip_match=True
    )
    cerebro.broker.setcash(cfg["backtest"]["cash"])
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
    sharpe = (strat.analyzers.sharpe.get_analysis() or {}).get("sharperatio", 0.0) or 0.0
    drawdown = strat.analyzers.drawdown.get_analysis() or {}
    trade_stats = strat.analyzers.trades.get_analysis() or {}
    return {
        "sharpe": sharpe,
        "drawdown": drawdown,
        "trades": trade_stats,
        "real_pnl": float(sum(real_trade_pnls)) if real_trade_pnls else 0.0,
    }


def _run_core(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, Dict, bt.Cerebro]:
    """Single core path used by both public entry points."""
    data = PrecomputedData(dataname=df)
    cerebro = _build_cerebro(cfg, data)
    results = _run_quiet(cerebro, cfg)
    strat = results[0]
    metrics_df = strat.get_metrics()
    stats = _collect_stats(strat)
    return metrics_df, stats, cerebro


# ---------- public API (thin wrappers; zero drift) ----------

def run_backtest(config_or_path: str | dict = "config.yml"):
    """Convenience: load data by path from config (prefers Parquet), then run."""
    cfg = config_or_path if isinstance(config_or_path, dict) else load_config(config_or_path)
    df = _load_df_from_cfg(cfg)
    return _run_core(df, cfg)


def run_backtest_df(df: pd.DataFrame, cfg: dict):
    """Fast path for sweeps: caller passes an in-memory DataFrame (no per-run I/O)."""
    dfx = _sanitize_df(df, cfg)
    return _run_core(dfx, cfg)
