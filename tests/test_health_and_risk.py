"""Behavioral tests for:
  - Health check aggregation (Stage A4)
  - Paper-book risk metrics: Sortino, Calmar, max-DD (Stage A3)
"""
from __future__ import annotations
import math
from pathlib import Path
import duckdb
import pytest


# ── Shared fixture: fresh live-signals DB ──────────────────────────────────

@pytest.fixture
def live_db(tmp_path, monkeypatch):
    db = tmp_path / "live-signals.duckdb"
    con = duckdb.connect(str(db))
    con.execute("""
        CREATE TABLE live_signals (
            signal_id VARCHAR PRIMARY KEY, strategy_id VARCHAR NOT NULL,
            fired_at TIMESTAMP, bar_timestamp TIMESTAMP, symbol VARCHAR,
            timeframe VARCHAR, signal_type VARCHAR, pattern VARCHAR,
            direction VARCHAR, ftfc_aligned BOOLEAN,
            entry_price DOUBLE, stop_price DOUBLE, target_price DOUBLE,
            recommended_size DOUBLE, notification_sent BOOLEAN DEFAULT FALSE,
            notification_channel VARCHAR, status VARCHAR DEFAULT 'new',
            executed_at TIMESTAMP, executed_price DOUBLE, exit_at TIMESTAMP,
            exit_price DOUBLE, exit_reason VARCHAR, realized_return DOUBLE,
            engine_version VARCHAR, metadata VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE paper_positions (
            position_id VARCHAR PRIMARY KEY, signal_id VARCHAR, strategy_id VARCHAR,
            symbol VARCHAR, timeframe VARCHAR, direction VARCHAR,
            opened_at TIMESTAMP, opened_price DOUBLE, target_price DOUBLE, stop_price DOUBLE,
            size_fraction DOUBLE, notional_size DOUBLE, status VARCHAR DEFAULT 'open',
            closed_at TIMESTAMP, closed_price DOUBLE, realized_pct_return DOUBLE, realized_pnl_usd DOUBLE,
            holding_bars INTEGER, last_mtm_at TIMESTAMP, last_mtm_price DOUBLE,
            unrealized_pct_return DOUBLE, metadata VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE paper_nav_snapshots (
            snap_date DATE PRIMARY KEY, nav_usd DOUBLE,
            n_open INTEGER, n_closed_today INTEGER, realized_today DOUBLE, unrealized DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE detect_errors (
            error_id VARCHAR PRIMARY KEY, run_id VARCHAR, strategy_id VARCHAR NOT NULL,
            error_at TIMESTAMP NOT NULL, error_type VARCHAR NOT NULL,
            error_message VARCHAR NOT NULL, traceback_text VARCHAR, engine_version VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE notification_log (
            notif_id VARCHAR PRIMARY KEY, signal_id VARCHAR, channel VARCHAR NOT NULL,
            attempted_at TIMESTAMP NOT NULL, attempts INTEGER NOT NULL DEFAULT 1,
            status VARCHAR NOT NULL, error_message VARCHAR, elapsed_ms INTEGER
        )
    """)
    con.close()

    from strategy_engine.paper import reporting as rep
    from strategy_engine.live import health, detector, notification
    monkeypatch.setattr(rep, "LIVE_DB", db)
    monkeypatch.setattr(detector, "LIVE_DB", db)
    monkeypatch.setattr(notification, "LIVE_DB", db)
    monkeypatch.setattr(health, "LIVE_DB", db)
    yield db


# ── A3: Risk metrics (Sortino, Calmar, max-DD) ──────────────────────────────

def test_risk_metrics_insufficient_snapshots(live_db):
    from strategy_engine.paper.reporting import nav_risk_metrics
    # 0 snapshots → everything zero, n_snapshots=0
    m = nav_risk_metrics()
    assert m.n_snapshots == 0
    assert m.sortino_ratio == 0.0


def test_risk_metrics_with_multiple_down_days(live_db):
    """Need ≥2 down days for Sortino to compute (single-sample std is 0)."""
    from strategy_engine.paper.reporting import nav_risk_metrics
    con = duckdb.connect(str(live_db))
    try:
        con.execute("DELETE FROM paper_nav_snapshots")
        for d, nav in [
            ("2026-04-15", 100_000),
            ("2026-04-16", 102_000),   # +2%
            ("2026-04-17", 100_500),   # -1.5%  (down 1)
            ("2026-04-18", 104_000),   # +3.5%
            ("2026-04-21", 103_000),   # -0.96% (down 2)
            ("2026-04-22", 105_500),   # +2.4%
            ("2026-04-23", 110_000),   # +4.3%
        ]:
            con.execute(
                "INSERT INTO paper_nav_snapshots VALUES (?, ?, 0, 0, 0, 0)",
                [d, nav],
            )
    finally:
        con.close()

    m = nav_risk_metrics()
    assert m.n_snapshots == 7
    assert m.total_return_pct == pytest.approx(0.10, abs=1e-6)
    assert m.max_drawdown_pct < 0                  # at least one drawdown
    assert m.sortino_ratio > 0                     # downside std > 0 now
    assert m.calmar_ratio > 0
    assert m.best_day_pct > 0
    assert m.worst_day_pct < 0


def test_strategy_equity_curves_accumulate_pnl(live_db):
    from strategy_engine.paper.reporting import strategy_equity_curves
    con = duckdb.connect(str(live_db))
    try:
        # Two closed positions for 'strat-a', one for 'strat-b'
        for pid, sid, closed_at, pnl in [
            ("p1", "strat-a", "2026-04-20 17:00", 500.0),
            ("p2", "strat-a", "2026-04-21 17:00", -200.0),
            ("p3", "strat-b", "2026-04-22 17:00", 300.0),
        ]:
            con.execute(
                "INSERT INTO paper_positions (position_id, strategy_id, status, closed_at, realized_pnl_usd) "
                "VALUES (?, ?, 'closed-target', ?::TIMESTAMP, ?)",
                [pid, sid, closed_at, pnl],
            )
    finally:
        con.close()

    curves = strategy_equity_curves()
    assert set(curves.keys()) == {"strat-a", "strat-b"}
    assert curves["strat-a"] == [
        ("2026-04-20 17:00:00", 500.0),
        ("2026-04-21 17:00:00", 300.0),
    ]
    assert curves["strat-b"] == [("2026-04-22 17:00:00", 300.0)]


def test_export_equity_png_creates_file(live_db, tmp_path):
    from strategy_engine.paper.reporting import export_equity_png
    con = duckdb.connect(str(live_db))
    try:
        con.execute(
            "INSERT INTO paper_positions (position_id, strategy_id, status, closed_at, realized_pnl_usd) "
            "VALUES (?, ?, 'closed-target', ?::TIMESTAMP, ?)",
            ["p1", "strat-a", "2026-04-20 17:00", 500.0],
        )
    finally:
        con.close()

    out = export_equity_png("strat-a", output_dir=tmp_path)
    assert out is not None
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial PNG


def test_export_equity_png_returns_none_for_unknown_strategy(live_db, tmp_path):
    from strategy_engine.paper.reporting import export_equity_png
    assert export_equity_png("does-not-exist", output_dir=tmp_path) is None


# ── A4: Health check ────────────────────────────────────────────────────────

def test_health_check_databases_all_green(live_db, monkeypatch, tmp_path):
    from strategy_engine.live import health

    # Isolate other DB paths we check against tmp
    fake_backtest = tmp_path / "backtest-results.duckdb"
    fake_ticks = tmp_path / "live-ticks.duckdb"
    con = duckdb.connect(str(fake_backtest))
    con.execute("CREATE TABLE backtest_results (id INTEGER)")
    con.close()
    con = duckdb.connect(str(fake_ticks))
    con.execute("CREATE TABLE ohlcv (symbol VARCHAR)")
    con.execute("CREATE TABLE poll_log (poll_id VARCHAR)")
    con.close()
    monkeypatch.setattr(health, "BACKTEST_DB", fake_backtest)
    monkeypatch.setattr(health, "LIVE_TICKS_DB", fake_ticks)

    checks = health.check_databases()
    severities = [c.severity for c in checks]
    assert all(s == "ok" for s in severities), severities


def test_health_check_detects_missing_table(live_db, tmp_path, monkeypatch):
    from strategy_engine.live import health
    # Replace one of the expected tables with a wrong name
    bad = tmp_path / "broken.duckdb"
    con = duckdb.connect(str(bad))
    con.execute("CREATE TABLE wrong_name (x INTEGER)")
    con.close()
    check = health._check_db(bad, ["expected_table"])
    assert check.severity == "error"
    assert "missing tables" in check.detail


def test_health_check_launchd_agents(monkeypatch):
    from strategy_engine.live import health

    # Fake launchctl output with partial expected agents loaded
    fake_loaded = {
        "com.matias.strategy-engine.paper-mtm",
        "com.matias.rotate-logs",
        # Others missing
    }
    checks = health.check_launchd_agents(loaded_fn=lambda: fake_loaded)
    severities = {c.name: c.severity for c in checks}
    assert severities["launchd:com.matias.strategy-engine.paper-mtm"] == "ok"
    assert severities["launchd:com.matias.rotate-logs"] == "ok"
    # Missing ones are warn (not error — they may be intentionally off)
    assert severities["launchd:com.matias.strategy-engine.daily-detect"] == "warn"


def test_health_check_detects_recent_errors(live_db):
    from strategy_engine.live import health
    from strategy_engine.live.detector import _log_detect_error

    # Threshold is 5 — we insert 6
    for i in range(6):
        try: raise ValueError(f"err-{i}")
        except ValueError as e: _log_detect_error("run-x", f"s-{i}", e)

    check = health.check_recent_detect_errors(hours=24, threshold=5)
    assert check.severity == "warn"
    assert "6" in check.detail


def test_health_check_paper_book_ok(live_db):
    from strategy_engine.live import health
    con = duckdb.connect(str(live_db))
    try:
        con.execute("INSERT INTO paper_nav_snapshots VALUES (CURRENT_DATE, 100000, 0, 0, 0, 0)")
    finally:
        con.close()
    check = health.check_paper_book_invariants()
    assert check.severity == "ok"


def test_health_check_paper_book_flags_stale_open(live_db):
    from strategy_engine.live import health
    con = duckdb.connect(str(live_db))
    try:
        con.execute("INSERT INTO paper_nav_snapshots VALUES (CURRENT_DATE, 100000, 0, 0, 0, 0)")
        # An open position from 20 weeks ago (should have been closed on the forward-window)
        con.execute(
            "INSERT INTO paper_positions (position_id, strategy_id, status, opened_at) "
            "VALUES ('zombie', 'any', 'open', CURRENT_TIMESTAMP - INTERVAL 20 WEEK)"
        )
    finally:
        con.close()
    check = health.check_paper_book_invariants()
    assert check.severity == "warn"
    assert "stale_open" in check.extras
    assert check.extras["stale_open"] == 1


def test_health_aggregate_status(monkeypatch):
    """If any check is 'error', the aggregate must be 'error'."""
    from strategy_engine.live import health

    monkeypatch.setattr(health, "check_databases",
                         lambda: [health.Check("db:x", "ok", "ok")])
    monkeypatch.setattr(health, "check_launchd_agents",
                         lambda *a, **k: [health.Check("launchd:y", "warn", "missing")])
    monkeypatch.setattr(health, "check_recent_detect_errors",
                         lambda: health.Check("errors", "ok", "ok"))
    monkeypatch.setattr(health, "check_notification_health",
                         lambda: health.Check("notif", "ok", "ok"))
    monkeypatch.setattr(health, "check_paper_book_invariants",
                         lambda: health.Check("paper", "error", "NAV=-100"))
    monkeypatch.setattr(health, "check_registry",
                         lambda: health.Check("reg", "ok", "ok"))
    monkeypatch.setattr(health, "check_promoted_dispatch",
                         lambda: [health.Check("dispatch", "ok", "ok")])

    report = health.run_health_check()
    assert report.status == "error"
    assert "error" in report.summary_line().lower()
