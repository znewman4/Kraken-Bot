# src/data_cleaning.py
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------ Helpers -----------------------------------

def _qc_dir() -> Path:
    d = Path("logs/data_quality")
    d.mkdir(parents=True, exist_ok=True)
    return d

def _export(df: pd.DataFrame, fname: str, enabled: bool):
    if enabled and not df.empty:
        df.to_csv(_qc_dir() / fname, index=False)

def _ensure_time(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        out = df.copy()
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
        return out.sort_values("time")
    out = df.reset_index().rename(columns={"index": "time"}).copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    return out.sort_values("time")

# --------------------------- Cleaning -------------------------------------

def clean_ohlcv(df: pd.DataFrame, data_cfg: dict | None = None) -> pd.DataFrame:
    """
    Minimal safe cleaning:
      - ensure time column, sort ascending
      - drop duplicate timestamps (keep first)
      - coerce numerics
      - optionally drop rows with impossible values
    """
    cfg = data_cfg or {}
    drop_bad_geometry = bool(cfg.get("drop_bad_geometry", True))
    drop_negative_volume = bool(cfg.get("drop_negative_volume", True))

    df = _ensure_time(df)
    df = df.drop_duplicates(subset=["time"], keep="first")

    # coerce numerics
    for col in ["open","high","low","close","volume","vwap","count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # drop bad geometry
    if drop_bad_geometry and all(c in df.columns for c in ["open","high","low","close"]):
        mask = (df["high"] < df["low"]) | \
               (df["open"] > df["high"]) | (df["open"] < df["low"]) | \
               (df["close"] > df["high"]) | (df["close"] < df["low"])
        n = int(mask.sum())
        if n:
            logger.warning("[CLEAN] Dropping %d rows with impossible price geometry", n)
            df = df[~mask]

    # drop negative volume
    if drop_negative_volume and "volume" in df.columns:
        mask = df["volume"] < 0
        n = int(mask.sum())
        if n:
            logger.warning("[CLEAN] Dropping %d rows with negative volume", n)
            df = df[~mask]


    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.set_index("time").sort_index()
    return df

# --------------------------- Validation -----------------------------------

def validate_ohlcv(
    df: pd.DataFrame,
    *,
    minutes: int = 5,
    label: str = "final",
    export_snippets: bool = True,
    jump_threshold_pct: float = 5.0,
) -> None:
    """
    QC scan with warnings only. Never mutates df.
    """
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    df = _ensure_time(df)

    # 1) duplicates
    dup_mask = df["time"].duplicated(keep="first")
    if dup_mask.any():
        logger.warning("[QC] %s: %d duplicate timestamps", label, int(dup_mask.sum()))
        _export(df.loc[dup_mask], f"duplicates_{label}_{ts}.csv", export_snippets)

    # 2) gaps
    dt_min = df["time"].diff().dt.total_seconds().div(60)
    gap_edges = df.loc[dt_min > minutes]
    if not gap_edges.empty:
        gaps = []
        for idx, row in gap_edges.iterrows():
            prev_time = df.loc[idx-1, "time"]
            this_time = row["time"]
            missing = int(dt_min.loc[idx] // minutes) - 1
            gaps.append([prev_time, this_time, missing])
        gap_df = pd.DataFrame(gaps, columns=["gap_start","gap_end","missing_bars"])
        total_missing = int(gap_df["missing_bars"].sum())
        span = (df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds() / 60.0
        coverage = 100.0 * (1 - total_missing / max(1, span/minutes))
        logger.warning("[QC] %s: %d gap edges, %d bars missing, coverage≈%.2f%%",
                       label, len(gap_df), total_missing, coverage)
        _export(gap_df, f"gaps_{label}_{ts}.csv", export_snippets)

    # 3) NaNs
    cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    nulls = df[cols].isna().sum()
    if int(nulls.sum()) > 0:
        logger.warning("[QC] %s: NaNs present — %s", label, dict(nulls))
        _export(df[df[cols].isna().any(axis=1)], f"nans_{label}_{ts}.csv", export_snippets)

    # 4) price geometry
    if all(c in df.columns for c in ["open","high","low","close"]):
        bad_hilo = df["high"] < df["low"]
        bad_bounds = ((df["open"] > df["high"]) | (df["open"] < df["low"]) |
                      (df["close"] > df["high"]) | (df["close"] < df["low"]))
        if bad_hilo.any() or bad_bounds.any():
            logger.warning("[QC] %s: bad geometry rows=%d", label, int((bad_hilo|bad_bounds).sum()))
            _export(df[bad_hilo|bad_bounds], f"price_geometry_{label}_{ts}.csv", export_snippets)

    # 5) volume anomalies
    if "volume" in df.columns:
        neg_vol = int((df["volume"] < 0).sum())
        if neg_vol:
            logger.warning("[QC] %s: %d rows with negative volume", label, neg_vol)
        # long runs of zero-volume
        zero_mask = (df["volume"] == 0).astype(int)
        groups = (df["volume"] != 0).astype(int).cumsum()
        zero_run = zero_mask.groupby(groups).transform("size").where(zero_mask==1,0).max()
        if zero_run and int(zero_run) >= 3:
            logger.warning("[QC] %s: long run of zero-volume bars (max=%d)", label, int(zero_run))

    # 6) extreme jumps
    if "close" in df.columns:
        pct = df["close"].pct_change().abs()
        jump_mask = pct > (jump_threshold_pct/100.0)
        if jump_mask.any():
            logger.warning("[QC] %s: %d extreme jumps (>%s%%)", label, int(jump_mask.sum()), jump_threshold_pct)
            _export(df.loc[jump_mask, ["time","close"]],
                    f"jumps_{label}_{ts}.csv", export_snippets)
