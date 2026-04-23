"""Behavioral tests for the paper-trading layer."""
from __future__ import annotations
from datetime import datetime, date as date_type
from pathlib import Path
import pytest

import duckdb


@pytest.fixture
def paper_db(tmp_path, monkeypatch):
    """Point the paper modules at a fresh tmp DuckDB."""
    db_path = tmp_path / "live-signals.duckdb"

    from strategy_engine.paper import book, reporting
    monkeypatch.setattr(book, "LIVE_DB", db_path)
    monkeypatch.setattr(reporting, "LIVE_DB", db_path)

    # Create tables
    con = duckdb.connect(str(db_path))
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
    con.execute(
        "INSERT INTO paper_nav_snapshots VALUES (current_date, 100000, 0, 0, 0, 0)"
    )
    con.close()

    yield db_path


def test_open_position_from_signal_bullish(paper_db):
    from strategy_engine.paper import open_position_from_signal

    sig = {
        "signal_id": "sig-test-1",
        "strategy_id": "test-strat",
        "symbol": "SPY",
        "timeframe": "1d",
        "direction": "bullish",
        "fired_at": "2026-04-22T10:00:00-04:00",
        "entry_price": 500.0,
        "target_price": 525.0,
        "stop_price": 490.0,
        "recommended_size": 0.10,
        "bar_timestamp": "2026-04-22",
    }
    pos_id = open_position_from_signal(sig)
    assert pos_id == "pp-sig-test-1"
    # Dedup: second open with same signal_id returns None
    assert open_position_from_signal(sig) is None


def test_positions_listing(paper_db):
    from strategy_engine.paper import open_position_from_signal, list_positions

    for i, sym in enumerate(["SPY", "QQQ"]):
        open_position_from_signal({
            "signal_id": f"sig-{i}", "strategy_id": "test", "symbol": sym,
            "timeframe": "1d", "direction": "bullish",
            "entry_price": 100.0 + i, "target_price": 110.0 + i, "stop_price": 95.0 + i,
            "recommended_size": 0.10,
            "fired_at": "2026-04-22T10:00:00-04:00", "bar_timestamp": "2026-04-22",
        })
    open_pos = list_positions(status="open")
    assert len(open_pos) == 2
    closed = list_positions(status="closed-target")
    assert closed == []


def test_manual_close_records_pnl(paper_db):
    from strategy_engine.paper import open_position_from_signal, close_position, list_positions

    open_position_from_signal({
        "signal_id": "sig-1", "strategy_id": "test", "symbol": "SPY",
        "timeframe": "1d", "direction": "bullish",
        "entry_price": 100.0, "target_price": 110.0, "stop_price": 95.0,
        "recommended_size": 0.10,
        "fired_at": "2026-04-22T10:00:00-04:00", "bar_timestamp": "2026-04-22",
    })
    close_position("pp-sig-1", reason="closed-manual", price=105.0)
    closed = list_positions(status="closed-manual")
    assert len(closed) == 1
    pos = closed[0]
    assert pos["realized_pct_return"] == pytest.approx(0.05)
    assert pos["realized_pnl_usd"] == pytest.approx(100000 * 0.10 * 0.05)  # $500


def test_bearish_pnl_sign_flip(paper_db):
    from strategy_engine.paper import open_position_from_signal, close_position, list_positions

    open_position_from_signal({
        "signal_id": "sig-bear", "strategy_id": "test", "symbol": "TSLA",
        "timeframe": "1h", "direction": "bearish",
        "entry_price": 200.0, "target_price": 180.0, "stop_price": 210.0,
        "recommended_size": 0.05,
        "fired_at": "2026-04-22T10:00:00-04:00", "bar_timestamp": "2026-04-22",
    })
    # Close at 190 = bearish win (price fell)
    close_position("pp-sig-bear", reason="closed-target", price=190.0)
    closed = list_positions(status="closed-target")
    pos = closed[0]
    # Price moved -5%; bearish position gains +5%
    assert pos["realized_pct_return"] == pytest.approx(0.05)
    assert pos["realized_pnl_usd"] == pytest.approx(100000 * 0.05 * 0.05)


def test_overall_summary_counts(paper_db):
    from strategy_engine.paper import (
        open_position_from_signal, close_position, overall_summary,
    )

    for i in range(3):
        open_position_from_signal({
            "signal_id": f"sig-{i}", "strategy_id": "s", "symbol": "SPY",
            "timeframe": "1d", "direction": "bullish",
            "entry_price": 100.0, "target_price": 110.0, "stop_price": 95.0,
            "recommended_size": 0.10,
            "fired_at": "2026-04-22T10:00:00-04:00", "bar_timestamp": "2026-04-22",
        })
    close_position("pp-sig-0", reason="closed-target", price=110.0)
    close_position("pp-sig-1", reason="closed-stop", price=95.0)

    summary = overall_summary()
    assert summary["total"] == 3
    assert summary["n_open"] == 1
    assert summary["n_closed"] == 2
