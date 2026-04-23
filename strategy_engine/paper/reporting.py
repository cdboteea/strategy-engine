"""Paper-trading reporting queries + Phase-2 risk metrics & equity curves."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import math
import duckdb


LIVE_DB = Path.home() / "clawd" / "data" / "live-signals.duckdb"
REPORTS_DIR = Path.home() / "clawd" / "reports" / "paper"


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


# ── Phase 2: risk metrics + equity curves ──────────────────────────────────

@dataclass
class RiskMetrics:
    """Portfolio-level risk-adjusted metrics computed from the NAV curve."""
    n_snapshots: int = 0
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0         # e.g. -0.05 = 5% peak-to-trough
    sortino_ratio: float = 0.0            # annualized; penalizes only downside
    calmar_ratio: float = 0.0             # ann_return / abs(max_dd)
    best_day_pct: float = 0.0
    worst_day_pct: float = 0.0
    days_tracked: int = 0


def _compute_nav_metrics(navs: list[tuple]) -> RiskMetrics:
    """Compute RiskMetrics from a list of (date, nav_usd) tuples, sorted ascending."""
    if len(navs) < 2:
        return RiskMetrics(n_snapshots=len(navs))

    import pandas as pd
    df = pd.DataFrame(navs, columns=["date", "nav"])
    df["nav"] = df["nav"].astype(float)
    df = df.sort_values("date").reset_index(drop=True)

    daily_returns = df["nav"].pct_change().dropna()
    if daily_returns.empty:
        return RiskMetrics(n_snapshots=len(navs))

    total_return = float(df["nav"].iloc[-1] / df["nav"].iloc[0] - 1)
    days = (pd.to_datetime(df["date"].iloc[-1]) - pd.to_datetime(df["date"].iloc[0])).days or 1
    years = max(days / 365.25, 1e-6)
    ann_return = float((df["nav"].iloc[-1] / df["nav"].iloc[0]) ** (1 / years) - 1)

    running_max = df["nav"].cummax()
    dd = (df["nav"] - running_max) / running_max
    max_dd = float(dd.min()) if len(dd) else 0.0

    # Sortino — annualized by trading-day count (252)
    downside = daily_returns[daily_returns < 0]
    if len(downside) >= 2 and downside.std(ddof=0) > 0:
        sortino = float(daily_returns.mean() / downside.std(ddof=0) * math.sqrt(252))
    else:
        sortino = 0.0

    # Calmar: ann_return / abs(max_dd)
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0

    return RiskMetrics(
        n_snapshots=len(navs),
        total_return_pct=total_return,
        annualized_return_pct=ann_return,
        max_drawdown_pct=max_dd,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        best_day_pct=float(daily_returns.max()),
        worst_day_pct=float(daily_returns.min()),
        days_tracked=days,
    )


def nav_risk_metrics() -> RiskMetrics:
    """Compute Sortino / Calmar / max-DD on the full paper_nav_snapshots series."""
    con = duckdb.connect(str(LIVE_DB), read_only=True)
    try:
        rows = con.execute(
            "SELECT snap_date, nav_usd FROM paper_nav_snapshots ORDER BY snap_date"
        ).fetchall()
    finally:
        con.close()
    return _compute_nav_metrics(rows)


def strategy_equity_curves() -> dict[str, list[tuple[str, float]]]:
    """
    Build a simple per-strategy cumulative-pnl curve from closed positions.

    For each strategy with closed positions, we sort by closed_at and
    accumulate realized_pnl_usd over time. This is a rough "strategy
    equity curve" — doesn't account for unrealized or for NAV scaling,
    but it answers 'is this strategy still winning?' visually.

    Returns: {strategy_id: [(closed_at_date, cumulative_pnl_usd), ...]}
    """
    con = duckdb.connect(str(LIVE_DB), read_only=True)
    try:
        rows = con.execute(
            "SELECT strategy_id, closed_at, realized_pnl_usd "
            "FROM paper_positions "
            "WHERE status != 'open' AND closed_at IS NOT NULL "
            "ORDER BY strategy_id, closed_at"
        ).fetchall()
    finally:
        con.close()

    out: dict[str, list[tuple[str, float]]] = {}
    for sid, closed_at, pnl in rows:
        pnl = float(pnl or 0.0)
        series = out.setdefault(sid, [])
        prev = series[-1][1] if series else 0.0
        series.append((str(closed_at), prev + pnl))
    return out


def export_equity_png(strategy_id: str, output_dir: Path | None = None) -> Path | None:
    """Render a per-strategy cumulative-P&L PNG. Returns path, or None if no data.

    Uses matplotlib with Agg backend so it works headless.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = strategy_equity_curves()
    if strategy_id not in curves or not curves[strategy_id]:
        return None

    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    dates, pnls = zip(*curves[strategy_id])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dates, pnls, marker="o", color="#1a4480", linewidth=1.5)
    ax.axhline(0, color="#aaa", linewidth=0.6)
    ax.set_title(f"Paper equity curve — {strategy_id}")
    ax.set_ylabel("Cumulative realized P&L ($)")
    ax.set_xlabel("Closed-at date")
    ax.grid(True, alpha=0.3)
    # Rotate x-labels for readability
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    fig.tight_layout()

    out = output_dir / f"equity-{strategy_id}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def export_all_equity_pngs(output_dir: Path | None = None) -> list[Path]:
    """Render PNGs for every strategy that has at least one closed position."""
    curves = strategy_equity_curves()
    paths: list[Path] = []
    for sid in curves:
        p = export_equity_png(sid, output_dir=output_dir)
        if p:
            paths.append(p)
    return paths
