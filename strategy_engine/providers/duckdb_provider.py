"""
DuckDB OHLCV provider.

Primary source: ~/clawd/data/firstrate.duckdb (14.18B rows, 26-year history,
monthly refresh, last-bar ceiling ~3-5 weeks behind today — expected).

Fallback for recent daily data: ~/clawd/data/fmp.duckdb daily_prices (fresh
through yesterday).

Registry timeframes ↔ firstrate timeframes:
    1m  → 1min
    5m  → 5min
    30m → 30min
    1h  → 1hour
    1d  → day
    4h  → resampled from 1hour
    1w  → resampled from day
    1mo → resampled from day
"""
from __future__ import annotations
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from ..config import FIRSTRATE_DB, FMP_DB


# Registry name → firstrate name
_FIRSTRATE_TIMEFRAME = {
    "1m": "1min",
    "5m": "5min",
    "30m": "30min",
    "1h": "1hour",
    "1d": "day",
}

# Timeframes that must be resampled from a finer source
_RESAMPLE_FROM = {
    "4h": ("1hour", "4h"),     # resample 1hour → 4h
    "1w": ("day", "W"),        # resample day → weekly
    "1mo": ("day", "ME"),      # resample day → monthly (month-end)
}


class DataNotAvailable(Exception):
    """Raised when the provider cannot satisfy a request."""


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample a bar DataFrame (DatetimeIndex + OHLCV) to a coarser frequency."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    res = df.resample(rule).agg(agg).dropna(subset=["open"])
    return res


def load_ohlcv(
    symbol: str,
    timeframe: str,
    start: Optional[str | date] = None,
    end: Optional[str | date] = None,
    *,
    prefer_fmp_for_recent: bool = True,
) -> pd.DataFrame:
    """
    Load OHLCV for (symbol, timeframe) from firstrate; resample if needed.
    Returns a DataFrame with DatetimeIndex and columns [open, high, low, close, volume].

    Raises DataNotAvailable if no bars found.
    """
    if timeframe in _FIRSTRATE_TIMEFRAME:
        fr_tf = _FIRSTRATE_TIMEFRAME[timeframe]
        base_tf_is_day = (fr_tf == "day")
        df = _load_firstrate(symbol, fr_tf, start, end)
        # For daily data, splice in fresh fmp daily (fills the 3-5 week firstrate lag).
        if prefer_fmp_for_recent and base_tf_is_day and not df.empty:
            fmp_recent = _load_fmp_daily(symbol, start=df.index.max().date())
            if not fmp_recent.empty:
                fmp_recent = fmp_recent[fmp_recent.index > df.index.max()]
                if not fmp_recent.empty:
                    df = pd.concat([df, fmp_recent]).sort_index()
    elif timeframe in _RESAMPLE_FROM:
        base_tf, rule = _RESAMPLE_FROM[timeframe]
        base = _load_firstrate(symbol, base_tf, start, end)
        # If we're resampling from daily bars, splice fresh fmp daily into the
        # base BEFORE resampling — so weekly/monthly bars include latest data.
        if prefer_fmp_for_recent and base_tf == "day" and not base.empty:
            fmp_recent = _load_fmp_daily(symbol, start=base.index.max().date())
            if not fmp_recent.empty:
                fmp_recent = fmp_recent[fmp_recent.index > base.index.max()]
                if not fmp_recent.empty:
                    base = pd.concat([base, fmp_recent]).sort_index()
        df = _resample(base, rule)
    else:
        raise ValueError(f"Unsupported timeframe {timeframe!r}")

    if df.empty:
        raise DataNotAvailable(f"No firstrate data for {symbol} {timeframe} in [{start}, {end}]")

    return df


def _load_firstrate(
    symbol: str,
    tf_name: str,
    start: Optional[str | date],
    end: Optional[str | date],
) -> pd.DataFrame:
    con = duckdb.connect(str(FIRSTRATE_DB), read_only=True)
    try:
        sql = """
            SELECT datetime, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """
        params: list = [symbol, tf_name]
        if start is not None:
            sql += " AND datetime >= ?"
            params.append(str(start))
        if end is not None:
            sql += " AND datetime <= ?"
            params.append(str(end))
        sql += " ORDER BY datetime"
        df = con.execute(sql, params).df()
    finally:
        con.close()

    if df.empty:
        return df
    df = df.set_index("datetime")
    return df


def _load_fmp_daily(
    symbol: str,
    start: Optional[str | date] = None,
    end: Optional[str | date] = None,
) -> pd.DataFrame:
    """Load daily bars from fmp.daily_prices (recent data, fresh through yesterday).

    Graceful-fail: returns empty DataFrame if the DB is locked by another
    process (e.g. the daily_update.py cron). For historical backtests the
    firstrate data through ~2026-03-27 is sufficient; the FMP splice only
    adds the most-recent-weeks for live signal detection.
    """
    if not Path(FMP_DB).exists():
        return pd.DataFrame()
    try:
        con = duckdb.connect(str(FMP_DB), read_only=True)
    except duckdb.IOException:
        # Another process holds a conflicting lock — proceed without FMP splice.
        return pd.DataFrame()
    try:
        sql = """
            SELECT date::timestamp AS datetime, open, high, low, close, volume
            FROM daily_prices
            WHERE symbol = ?
        """
        params: list = [symbol]
        if start is not None:
            sql += " AND date >= ?"
            params.append(str(start))
        if end is not None:
            sql += " AND date <= ?"
            params.append(str(end))
        sql += " ORDER BY date"
        df = con.execute(sql, params).df()
    except duckdb.Error:
        return pd.DataFrame()
    finally:
        con.close()

    if df.empty:
        return df
    df = df.set_index("datetime")
    return df


def load_multi_timeframe(
    symbol: str,
    timeframes: list[str],
    start: Optional[str | date] = None,
    end: Optional[str | date] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load OHLCV bars for a symbol across multiple timeframes.

    Returns a dict {timeframe: DataFrame}. Timeframes that fail to load
    produce an empty DataFrame (caller decides how to handle).
    """
    out: dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        try:
            out[tf] = load_ohlcv(symbol, tf, start=start, end=end)
        except (DataNotAvailable, ValueError) as e:
            out[tf] = pd.DataFrame()
    return out


def describe_availability(symbol: str, timeframe: str) -> dict:
    """Diagnostics — what's available for this (symbol, timeframe)?"""
    info: dict = {"symbol": symbol, "timeframe": timeframe}
    if timeframe in _FIRSTRATE_TIMEFRAME:
        info["source"] = "firstrate"
        info["firstrate_name"] = _FIRSTRATE_TIMEFRAME[timeframe]
    elif timeframe in _RESAMPLE_FROM:
        base, rule = _RESAMPLE_FROM[timeframe]
        info["source"] = f"firstrate({base}) → resample({rule})"
    else:
        info["source"] = "unsupported"
        return info
    con = duckdb.connect(str(FIRSTRATE_DB), read_only=True)
    try:
        tf_name = _FIRSTRATE_TIMEFRAME.get(timeframe) or _RESAMPLE_FROM[timeframe][0]
        row = con.execute(
            """
            SELECT COUNT(*) AS bars, MIN(datetime)::date AS first_bar, MAX(datetime)::date AS last_bar
            FROM ohlcv WHERE symbol = ? AND timeframe = ?
            """,
            [symbol, tf_name],
        ).fetchone()
        info.update({"bars": row[0], "first_bar": row[1], "last_bar": row[2]})
    finally:
        con.close()
    return info
