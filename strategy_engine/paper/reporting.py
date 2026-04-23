"""Paper-trading reporting queries."""
from __future__ import annotations
from pathlib import Path
import duckdb


LIVE_DB = Path.home() / "clawd" / "data" / "live-signals.duckdb"


def list_positions(status: str = "open") -> list[dict]:
    """Return list of positions as dicts (filtered by status)."""
    con = duckdb.connect(str(LIVE_DB))
    try:
        sql = """
            SELECT position_id, strategy_id, symbol, timeframe, direction,
                   opened_at, opened_price, target_price, stop_price,
                   size_fraction, notional_size, status,
                   closed_at, closed_price, realized_pct_return, realized_pnl_usd,
                   last_mtm_at, last_mtm_price, unrealized_pct_return
            FROM paper_positions
        """
        params = []
        if status and status != "all":
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY opened_at DESC"
        cur = con.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()


def realized_pnl_by_strategy() -> list[dict]:
    """Per-strategy realized P&L summary."""
    con = duckdb.connect(str(LIVE_DB))
    try:
        cur = con.execute("""
            SELECT
              strategy_id,
              COUNT(*) AS n_positions,
              SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) AS n_open,
              SUM(CASE WHEN status != 'open' THEN 1 ELSE 0 END) AS n_closed,
              SUM(CASE WHEN status = 'closed-target' THEN 1 ELSE 0 END) AS n_target,
              SUM(CASE WHEN status = 'closed-stop' THEN 1 ELSE 0 END) AS n_stop,
              SUM(CASE WHEN status = 'closed-window' THEN 1 ELSE 0 END) AS n_window,
              COALESCE(SUM(realized_pnl_usd), 0) AS realized_usd,
              COALESCE(AVG(realized_pct_return), 0) AS avg_pct_return,
              COALESCE(AVG(CASE WHEN realized_pct_return > 0 THEN 1.0 ELSE 0.0 END), 0) AS win_rate
            FROM paper_positions
            GROUP BY strategy_id
            ORDER BY realized_usd DESC
        """)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()


def overall_summary() -> dict:
    """High-level book stats."""
    con = duckdb.connect(str(LIVE_DB))
    try:
        pos = con.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) AS n_open, "
            "SUM(CASE WHEN status!='open' THEN 1 ELSE 0 END) AS n_closed, "
            "COALESCE(SUM(realized_pnl_usd), 0) AS realized, "
            "COALESCE(SUM(notional_size * unrealized_pct_return), 0) AS unrealized_open_usd "
            "FROM paper_positions"
        ).fetchone()
        cols = ["total", "n_open", "n_closed", "realized_usd", "unrealized_usd"]
        base = dict(zip(cols, pos))

        nav = con.execute(
            "SELECT snap_date, nav_usd, n_open FROM paper_nav_snapshots ORDER BY snap_date DESC LIMIT 1"
        ).fetchone()
        if nav:
            base["latest_nav"] = float(nav[1])
            base["latest_nav_date"] = str(nav[0])
        return base
    finally:
        con.close()
