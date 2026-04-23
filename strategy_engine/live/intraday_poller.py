"""
EODHD intraday poller.

Fetches latest 1h (and 5m/15m/30m if needed) bars for every ticker appearing
in a PROMOTED strategy with an intraday timeframe, and upserts them into
~/clawd/data/live-ticks.duckdb.

The duckdb provider splices live-ticks into queries for those timeframes,
filling the gap between firstrate's monthly refresh ceiling and today's
market state.

Usage:
    from strategy_engine.live.intraday_poller import poll_promoted
    summary = poll_promoted()
    print(summary)

Or via CLI:
    strategy-engine intraday poll
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import time
import uuid

import duckdb
import pandas as pd

from ..registry.loader import load_all
from ..registry.schema import Strategy


LIVE_TICKS_DB = Path.home() / "clawd" / "data" / "live-ticks.duckdb"


# Timeframes that need intraday polling (others use firstrate+fmp daily splice).
# 4h is resampled from 1h, so polling 1h covers 4h too.
_INTRADAY_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1h", "4h"}

# Registry timeframe → EODHD interval (4h → 1h since 4h is resampled client-side)
_EODHD_INTERVAL = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "1h",
}

# Internal table timeframe labels (aligned with firstrate naming)
_TABLE_TIMEFRAME = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1hour",
    "4h": "1hour",  # store as 1hour; provider resamples to 4h
}


@dataclass
class PollTarget:
    symbol: str
    timeframe: str
    eodhd_interval: str
    table_timeframe: str


@dataclass
class PollResult:
    poll_id: str
    started_at: datetime
    finished_at: datetime | None = None
    targets: list[PollTarget] = field(default_factory=list)
    bars_inserted: int = 0
    bars_updated: int = 0
    n_ok: int = 0
    n_err: int = 0
    errors: list[str] = field(default_factory=list)
    status: str = "running"


def collect_targets(strategies: list[Strategy] | None = None) -> list[PollTarget]:
    """Determine which (symbol, timeframe) pairs need intraday polling.

    A target is added for any PROMOTED strategy with an intraday timeframe.
    We deduplicate across strategies so SPY appears at most once per
    timeframe even if 5 strategies want it.
    """
    strategies = strategies if strategies is not None else load_all()
    seen: set[tuple[str, str]] = set()
    targets: list[PollTarget] = []
    for s in strategies:
        if s.status not in ("promoted", "live-ready", "live"):
            continue
        tf = s.timeframe
        if tf not in _INTRADAY_TIMEFRAMES:
            continue
        for sym in s.instruments:
            key = (sym.upper(), tf)
            if key in seen:
                continue
            seen.add(key)
            targets.append(
                PollTarget(
                    symbol=sym.upper(),
                    timeframe=tf,
                    eodhd_interval=_EODHD_INTERVAL[tf],
                    table_timeframe=_TABLE_TIMEFRAME[tf],
                )
            )
    # Deduplicate again at the (symbol, table_timeframe) level — 4h and 1h
    # both target the 1hour table, so we only need one fetch per symbol.
    seen2: set[tuple[str, str]] = set()
    out: list[PollTarget] = []
    for t in targets:
        key = (t.symbol, t.eodhd_interval)
        if key in seen2:
            continue
        seen2.add(key)
        out.append(t)
    return out


def _latest_stored(con: duckdb.DuckDBPyConnection, symbol: str, table_tf: str) -> datetime | None:
    row = con.execute(
        "SELECT MAX(datetime) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
        [symbol, table_tf],
    ).fetchone()
    if row and row[0]:
        return row[0]
    return None


def _upsert_bars(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    table_tf: str,
    bars: list[dict],
) -> tuple[int, int]:
    """UPSERT bars into the ohlcv table. Returns (inserted, updated)."""
    if not bars:
        return 0, 0
    # Build a staging DataFrame
    df = pd.DataFrame(bars)
    # EODHD intraday returns: timestamp (unix), datetime (UTC string), gmtoffset, open, high, low, close, volume
    # Normalize to our schema
    df = df.rename(columns={"datetime": "datetime_str"})
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(None)
    df["symbol"] = symbol
    df["timeframe"] = table_tf
    df["source"] = "eodhd-intraday"
    df["fetched_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    out = df[
        ["symbol", "datetime", "open", "high", "low", "close", "volume",
         "timeframe", "source", "fetched_at"]
    ].copy()
    out = out.dropna(subset=["open", "high", "low", "close"])  # skip gap bars
    # Pre-count what's already stored for the incoming datetimes
    incoming = [ts for ts in out["datetime"].tolist()]
    if not incoming:
        return 0, 0
    # DELETE conflicting rows (our PK is symbol+timeframe+datetime+source),
    # then INSERT. Cheaper than per-row ON CONFLICT in duckdb.
    placeholders = ",".join("?" for _ in incoming)
    pre = con.execute(
        f"SELECT COUNT(*) FROM ohlcv "
        f"WHERE symbol = ? AND timeframe = ? AND source = 'eodhd-intraday' "
        f"AND datetime IN ({placeholders})",
        [symbol, table_tf, *incoming],
    ).fetchone()[0]
    con.execute(
        f"DELETE FROM ohlcv "
        f"WHERE symbol = ? AND timeframe = ? AND source = 'eodhd-intraday' "
        f"AND datetime IN ({placeholders})",
        [symbol, table_tf, *incoming],
    )
    con.register("staging_df", out)
    con.execute(
        """
        INSERT INTO ohlcv (symbol, datetime, open, high, low, close, volume,
                           timeframe, source, fetched_at)
        SELECT symbol, datetime, open, high, low, close, volume,
               timeframe, source, fetched_at
        FROM staging_df
        """
    )
    con.unregister("staging_df")
    total = len(out)
    updated = pre
    inserted = total - updated
    return inserted, updated


def poll_promoted(
    *,
    strategies: list[Strategy] | None = None,
    client=None,
    lookback_days: int = 5,
) -> PollResult:
    """
    Fetch fresh intraday bars for every promoted intraday strategy.

    `client` can be injected for testing; defaults to EODHDClient() which
    auto-loads key from Keychain.

    `lookback_days` controls how far back we always pull — gives us
    overlap with prior polls and resilience against gap days.
    """
    result = PollResult(
        poll_id=f"poll-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}",
        started_at=datetime.now(timezone.utc),
    )
    result.targets = collect_targets(strategies)

    if not result.targets:
        result.status = "ok-empty"
        result.finished_at = datetime.now(timezone.utc)
        _log_poll(result)
        return result

    if client is None:
        # Import lazily so tests can run without the fmp-toolkit package
        sys.path.insert(0, str(Path.home() / "projects" / "fmp-toolkit"))
        from eodhd_client import EODHDClient  # type: ignore
        client = EODHDClient()

    con = duckdb.connect(str(LIVE_TICKS_DB))
    try:
        now_ts = int(time.time())
        from_ts = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp())
        for t in result.targets:
            try:
                bars = client.get_intraday(
                    t.symbol, interval=t.eodhd_interval,
                    from_ts=from_ts, to_ts=now_ts,
                )
                ins, upd = _upsert_bars(con, t.symbol, t.table_timeframe, bars or [])
                result.bars_inserted += ins
                result.bars_updated += upd
                result.n_ok += 1
            except Exception as e:  # noqa: BLE001 — report all per-target failures
                result.n_err += 1
                result.errors.append(f"{t.symbol} {t.timeframe}: {e!s}")
    finally:
        con.close()

    result.finished_at = datetime.now(timezone.utc)
    result.status = "ok" if result.n_err == 0 else ("partial" if result.n_ok > 0 else "failed")
    _log_poll(result)
    return result


def _log_poll(result: PollResult) -> None:
    con = duckdb.connect(str(LIVE_TICKS_DB))
    try:
        con.execute(
            """
            INSERT INTO poll_log (
                poll_id, started_at, finished_at, symbols, timeframes,
                bars_inserted, bars_updated, n_symbols_ok, n_symbols_err,
                error_summary, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                result.poll_id,
                result.started_at,
                result.finished_at,
                ",".join(sorted({t.symbol for t in result.targets})),
                ",".join(sorted({t.table_timeframe for t in result.targets})),
                result.bars_inserted,
                result.bars_updated,
                result.n_ok,
                result.n_err,
                "; ".join(result.errors)[:2000] if result.errors else None,
                result.status,
            ],
        )
    finally:
        con.close()
