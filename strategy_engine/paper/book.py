"""
Paper-trading book operations.

On signal fire:
  → open_position_from_signal(sig) creates a paper position sized at
    capital_allocation × current_nav with the signal's entry/target/stop.

On bar close (via mark-to-market cron):
  → mark_to_market_all() walks all open positions, fetches the latest bar's
    high/low/close, and:
      - closes any position where target or stop was hit
      - updates unrealized_pct_return for positions still open
      - writes a paper_nav_snapshot row for today

On exit:
  → close_position(pos_id, reason, price, date) records the closure and
    updates realized_pct_return.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date as date_type
from pathlib import Path
from typing import Optional
import json
import uuid

import duckdb
import pandas as pd

from ..providers.duckdb_provider import load_ohlcv, DataNotAvailable
from ..registry.loader import load_one


LIVE_DB = Path.home() / "clawd" / "data" / "live-signals.duckdb"
INITIAL_NAV = 100_000.0


def _connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(LIVE_DB))


def current_nav() -> float:
    """Latest NAV snapshot value. Falls back to INITIAL_NAV if none exists."""
    con = _connect()
    try:
        row = con.execute(
            "SELECT nav_usd FROM paper_nav_snapshots ORDER BY snap_date DESC LIMIT 1"
        ).fetchone()
        return float(row[0]) if row else INITIAL_NAV
    finally:
        con.close()


def open_position_from_signal(signal_row: dict) -> Optional[str]:
    """
    Given a row from live_signals (as dict), open a matching paper position.
    Returns the new position_id or None if a position already exists for this signal.

    Position sizing: capital_allocation × current_nav. We use the signal's
    entry_price if present (STRAT), otherwise the bar's close (Bollinger).
    """
    con = _connect()
    try:
        signal_id = signal_row["signal_id"]
        # Dedup — one paper position per signal
        existing = con.execute(
            "SELECT position_id FROM paper_positions WHERE signal_id = ?", [signal_id]
        ).fetchone()
        if existing:
            return None

        pos_id = f"pp-{signal_id}"
        size_fraction = float(signal_row.get("recommended_size") or 0.10)
        nav = current_nav()
        notional = nav * size_fraction
        opened_at = signal_row.get("fired_at") or datetime.now().astimezone().isoformat(timespec="seconds")
        opened_price = float(signal_row.get("entry_price") or 0.0)

        if opened_price <= 0:
            # Fallback: use bar close price. Get it from the symbol's latest bar.
            try:
                bars = load_ohlcv(signal_row["symbol"], signal_row["timeframe"])
                opened_price = float(bars.iloc[-1]["close"])
            except DataNotAvailable:
                return None

        con.execute(
            """
            INSERT INTO paper_positions (
                position_id, signal_id, strategy_id, symbol, timeframe, direction,
                opened_at, opened_price, target_price, stop_price,
                size_fraction, notional_size, status, metadata
            ) VALUES (?, ?, ?, ?, ?, ?,
                     ?::TIMESTAMP, ?, ?, ?,
                     ?, ?, 'open', ?)
            """,
            [
                pos_id, signal_id,
                signal_row["strategy_id"], signal_row["symbol"], signal_row["timeframe"],
                signal_row["direction"], opened_at, opened_price,
                signal_row.get("target_price"), signal_row.get("stop_price"),
                size_fraction, notional,
                json.dumps({"signal_bar_timestamp": signal_row.get("bar_timestamp")}),
            ],
        )
        return pos_id
    finally:
        con.close()


def _get_strategy_forward_window(strategy_id: str) -> Optional[int]:
    """Lookup the strategy's forward-window or max-holding-bars from YAML."""
    from ..backtest.runner import _find_yaml_path
    yaml_path = _find_yaml_path(strategy_id)
    if not yaml_path:
        return None
    strat = load_one(yaml_path)
    # Bollinger uses exit.forward_window_weeks; STRAT uses signal_logic.max_holding_bars
    fw = getattr(strat.exit, "forward_window_weeks", None)
    if fw is not None:
        return int(fw)
    mhb = getattr(strat.signal_logic, "max_holding_bars", None)
    if mhb is not None:
        return int(mhb)
    return None


def mark_to_market_all() -> dict:
    """
    Walk all open positions, update MTM, close any that hit target/stop or
    exceeded forward window. Returns a summary dict.
    """
    con = _connect()
    summary = {"opened": 0, "mtm_updated": 0, "closed_target": 0, "closed_stop": 0, "closed_window": 0, "errors": 0}
    try:
        rows = con.execute(
            "SELECT position_id, signal_id, strategy_id, symbol, timeframe, direction, "
            "       opened_at, opened_price, target_price, stop_price, notional_size "
            "FROM paper_positions WHERE status = 'open'"
        ).fetchall()

        for r in rows:
            pos_id, sig_id, strat_id, symbol, tf, direction, opened_at, opened_price, \
                target_price, stop_price, notional = r
            try:
                bars = load_ohlcv(symbol, tf)
            except DataNotAvailable:
                summary["errors"] += 1
                continue
            if bars.empty:
                continue

            # Only look at bars AFTER opened_at
            try:
                opened_ts = pd.Timestamp(opened_at)
            except Exception:
                opened_ts = pd.Timestamp(str(opened_at))
            future = bars[bars.index > opened_ts]
            if future.empty:
                continue

            close_reason = None
            close_price = None
            close_date = None

            for ts, row in future.iterrows():
                fwd_high = float(row["high"])
                fwd_low = float(row["low"])
                fwd_close = float(row["close"])

                if direction == "bullish":
                    if stop_price and fwd_low <= stop_price:
                        close_reason = "closed-stop"
                        close_price = stop_price
                        close_date = ts
                        break
                    if target_price and fwd_high >= target_price:
                        close_reason = "closed-target"
                        close_price = target_price
                        close_date = ts
                        break
                else:  # bearish
                    if stop_price and fwd_high >= stop_price:
                        close_reason = "closed-stop"
                        close_price = stop_price
                        close_date = ts
                        break
                    if target_price and fwd_low <= target_price:
                        close_reason = "closed-target"
                        close_price = target_price
                        close_date = ts
                        break

            # Max-holding window check
            if close_reason is None:
                max_bars = _get_strategy_forward_window(strat_id)
                if max_bars and len(future) >= max_bars:
                    last_bar = future.iloc[max_bars - 1] if max_bars <= len(future) else future.iloc[-1]
                    close_date = future.index[min(max_bars - 1, len(future) - 1)]
                    close_price = float(last_bar["close"])
                    close_reason = "closed-window"

            latest = future.iloc[-1]
            latest_close = float(latest["close"])
            latest_date = future.index[-1]

            if close_reason is not None:
                _close_position_on_con(
                    con, pos_id, close_reason, close_price, close_date,
                    opened_price, direction, notional,
                )
                if close_reason == "closed-target":
                    summary["closed_target"] += 1
                elif close_reason == "closed-stop":
                    summary["closed_stop"] += 1
                else:
                    summary["closed_window"] += 1
            else:
                # Update MTM fields
                unrealized = (latest_close - opened_price) / opened_price
                if direction == "bearish":
                    unrealized = -unrealized
                con.execute(
                    "UPDATE paper_positions SET last_mtm_at = ?::TIMESTAMP, "
                    "last_mtm_price = ?, unrealized_pct_return = ? WHERE position_id = ?",
                    [str(latest_date), latest_close, float(unrealized), pos_id],
                )
                summary["mtm_updated"] += 1

        snapshot_nav(con=con)
    finally:
        con.close()
    return summary


def _close_position_on_con(
    con: duckdb.DuckDBPyConnection,
    pos_id: str,
    reason: str,
    close_price: float,
    close_date,
    opened_price: float,
    direction: str,
    notional: float,
) -> None:
    pct_return = (close_price - opened_price) / opened_price
    if direction == "bearish":
        pct_return = -pct_return
    realized_usd = notional * pct_return
    con.execute(
        """
        UPDATE paper_positions
        SET status = ?,
            closed_at = ?::TIMESTAMP,
            closed_price = ?,
            realized_pct_return = ?,
            realized_pnl_usd = ?,
            last_mtm_at = ?::TIMESTAMP,
            last_mtm_price = ?,
            unrealized_pct_return = 0.0
        WHERE position_id = ?
        """,
        [reason, str(close_date), close_price, float(pct_return), float(realized_usd),
         str(close_date), close_price, pos_id],
    )


def close_position(pos_id: str, reason: str, price: float, at_date: Optional[date_type] = None) -> None:
    """Manually close a position."""
    con = _connect()
    try:
        row = con.execute(
            "SELECT opened_price, direction, notional_size FROM paper_positions WHERE position_id = ?",
            [pos_id],
        ).fetchone()
        if not row:
            raise ValueError(f"No position {pos_id!r}")
        opened_price, direction, notional = row
        dt = datetime.combine(at_date or date_type.today(), datetime.min.time()).isoformat()
        _close_position_on_con(
            con, pos_id, reason, price, dt, opened_price, direction, notional,
        )
    finally:
        con.close()


def snapshot_nav(con: Optional[duckdb.DuckDBPyConnection] = None) -> dict:
    """Write today's NAV snapshot. Can be called as part of mark_to_market_all,
    or standalone end-of-day."""
    close_con = False
    if con is None:
        con = _connect()
        close_con = True
    try:
        # NAV = initial + cumulative realized_pnl + current unrealized (based on notional)
        # Simpler: initial + sum(realized_pnl_usd) + sum(notional × unrealized_pct_return)
        realized_row = con.execute(
            "SELECT COALESCE(SUM(realized_pnl_usd), 0) FROM paper_positions"
        ).fetchone()
        realized_total = float(realized_row[0]) if realized_row else 0.0

        unreal_row = con.execute(
            "SELECT COALESCE(SUM(notional_size * unrealized_pct_return), 0) "
            "FROM paper_positions WHERE status = 'open'"
        ).fetchone()
        unrealized_total = float(unreal_row[0]) if unreal_row else 0.0

        nav = INITIAL_NAV + realized_total + unrealized_total

        n_open = con.execute(
            "SELECT COUNT(*) FROM paper_positions WHERE status = 'open'"
        ).fetchone()[0]
        n_closed_today = con.execute(
            "SELECT COUNT(*) FROM paper_positions "
            "WHERE status != 'open' AND closed_at::date = current_date"
        ).fetchone()[0]
        realized_today = con.execute(
            "SELECT COALESCE(SUM(realized_pnl_usd), 0) FROM paper_positions "
            "WHERE closed_at::date = current_date"
        ).fetchone()[0]

        # Upsert today's snapshot
        con.execute(
            "DELETE FROM paper_nav_snapshots WHERE snap_date = current_date"
        )
        con.execute(
            "INSERT INTO paper_nav_snapshots (snap_date, nav_usd, n_open, n_closed_today, realized_today, unrealized) "
            "VALUES (current_date, ?, ?, ?, ?, ?)",
            [nav, n_open, n_closed_today, float(realized_today), unrealized_total],
        )
        return {
            "nav_usd": nav,
            "n_open": n_open,
            "n_closed_today": n_closed_today,
            "realized_today": float(realized_today),
            "unrealized": unrealized_total,
        }
    finally:
        if close_con:
            con.close()
