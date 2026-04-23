"""End-to-end smoke test — simulates a full weekday cycle.

Covers integration points that unit tests miss:
  1. Detect across multiple strategies → some fire, some don't
  2. A replay of detect against the SAME bar produces NO duplicate signals
     (dedup test — catches the bug A5 fixed)
  3. Paper positions open from fired signals
  4. A replay of paper-position opening is idempotent
  5. MTM closes positions that hit target
  6. Health check reports OK at the end

We synthesize OHLCV bars in-memory (no DB dependency on firstrate/fmp) and
inject them via the provider. The strategies are synthetic too — enough to
exercise the pipeline without depending on registry YAMLs.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
import duckdb
import pandas as pd
import pytest

from strategy_engine.registry.schema import Strategy
from strategy_engine.live.detector import SignalFired


# ── Synthetic data generator ─────────────────────────────────────────────

def _synth_bollinger_breakdown_bars(
    start: str = "2026-01-01",
    periods: int = 40,
) -> pd.DataFrame:
    """Build a daily bar series whose LAST bar closes below the 20-SMA − 2σ,
    guaranteeing `detect_signals` fires on the latest completed bar.

    The series is flat-ish around 100, then the very last bar crashes
    enough to break the lower band. (Earlier dips get absorbed into the
    rolling std, which then widens and masks the signal.)"""
    import numpy as np
    rng = pd.date_range(start=start, periods=periods, freq="D")
    rng_seed = np.random.default_rng(42)
    base = np.full(periods, 100.0) + rng_seed.normal(0, 0.5, periods)
    # Only the last bar crashes — keeps rolling std tight
    base[-1] = 92.0
    df = pd.DataFrame({
        "open": base - 0.2,
        "high": base + 0.4,
        "low": base - 0.4,
        "close": base,
        "volume": 1_000_000,
    }, index=rng)
    return df


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def e2e_db(tmp_path, monkeypatch):
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
    con.execute(
        "INSERT INTO paper_nav_snapshots VALUES (current_date, 100000, 0, 0, 0, 0)"
    )
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

    from strategy_engine.live import detector, notification, health
    from strategy_engine.paper import book, reporting
    monkeypatch.setattr(detector, "LIVE_DB", db)
    monkeypatch.setattr(notification, "LIVE_DB", db)
    monkeypatch.setattr(health, "LIVE_DB", db)
    monkeypatch.setattr(book, "LIVE_DB", db)
    monkeypatch.setattr(reporting, "LIVE_DB", db)
    yield db


@pytest.fixture
def fake_bars(monkeypatch):
    """Short-circuit load_ohlcv to return our synthetic breakdown series."""
    bars = _synth_bollinger_breakdown_bars()

    def _fake_load_ohlcv(symbol, timeframe, start=None, end=None, **_):
        return bars

    # Patch in both places it's imported
    monkeypatch.setattr(
        "strategy_engine.providers.duckdb_provider.load_ohlcv",
        _fake_load_ohlcv,
    )
    # detector imports load_ohlcv directly — patch there too
    monkeypatch.setattr(
        "strategy_engine.live.detector.load_ohlcv",
        _fake_load_ohlcv,
    )
    return bars


def _synthetic_bollinger_strategy() -> Strategy:
    return Strategy.model_validate({
        "id": "synth-spy-bol-v1",
        "name": "Synthetic SPY Bollinger",
        "status": "promoted",
        "asset_class": "equity-index",
        "instruments": ["SPY"],
        "timeframe": "1d",
        "signal_logic": {
            "type": "bollinger-mean-reversion",
            "lookback": 20, "std_dev": 2.0,
        },
        "entry": {
            "mode": "hybrid-50-50",
            "first_half": "at-signal-close",
            "second_half": {"depth": -0.05, "trigger": "threshold"},
        },
        "exit": {
            "mode": "profit-target",
            "target": 0.05,
            "forward_window_weeks": 13,
        },
        "capital_allocation": 0.10,
        "data_sources": ["firstrate"],
    })


# ── E2E checks ───────────────────────────────────────────────────────────

def test_e2e_signal_fires_persists_opens_paper(e2e_db, fake_bars, monkeypatch):
    """Full path: detect → signal fires → persist → paper position opens."""
    from strategy_engine.live.detector import (
        detect_signals_for_strategy, persist_signal, _detect_bollinger,
    )
    from strategy_engine.paper import open_position_from_signal, list_positions
    from dataclasses import asdict

    strat = _synthetic_bollinger_strategy()
    sig = _detect_bollinger(strat)
    assert sig is not None, "breakdown bar should trigger a signal"
    assert sig.signal_id.startswith("sig-synth-spy-bol-v1-")

    persist_signal(sig)
    pos_id = open_position_from_signal(asdict(sig))
    assert pos_id == f"pp-{sig.signal_id}"

    open_positions = list_positions(status="open")
    assert len(open_positions) == 1
    assert open_positions[0]["strategy_id"] == "synth-spy-bol-v1"


def test_e2e_replay_is_idempotent(e2e_db, fake_bars, monkeypatch):
    """Re-running detect on the SAME bar must NOT create duplicate rows.
    This is the A5 regression guard."""
    from strategy_engine.live.detector import persist_signal, _detect_bollinger
    from strategy_engine.paper import open_position_from_signal
    from dataclasses import asdict

    strat = _synthetic_bollinger_strategy()

    # First pass
    sig_a = _detect_bollinger(strat)
    persist_signal(sig_a)
    pos_a = open_position_from_signal(asdict(sig_a))

    # Second pass (simulates a cron restart or double-fire)
    sig_b = _detect_bollinger(strat)
    assert sig_b.signal_id == sig_a.signal_id, "signal_id must be deterministic"
    persist_signal(sig_b)                      # idempotent
    pos_b = open_position_from_signal(asdict(sig_b))
    assert pos_b is None, "paper book must dedup on signal_id"

    # Verify DB state
    con = duckdb.connect(str(e2e_db), read_only=True)
    try:
        n_signals = con.execute("SELECT COUNT(*) FROM live_signals").fetchone()[0]
        n_positions = con.execute("SELECT COUNT(*) FROM paper_positions").fetchone()[0]
    finally:
        con.close()
    assert n_signals == 1
    assert n_positions == 1


def test_e2e_multiple_detect_passes_over_same_day(e2e_db, fake_bars):
    """10× replay within the same session still produces exactly 1 row each."""
    from strategy_engine.live.detector import persist_signal, _detect_bollinger
    from strategy_engine.paper import open_position_from_signal
    from dataclasses import asdict

    strat = _synthetic_bollinger_strategy()
    for _ in range(10):
        sig = _detect_bollinger(strat)
        persist_signal(sig)
        open_position_from_signal(asdict(sig))

    con = duckdb.connect(str(e2e_db), read_only=True)
    try:
        n_sig = con.execute("SELECT COUNT(*) FROM live_signals").fetchone()[0]
        n_pos = con.execute("SELECT COUNT(*) FROM paper_positions").fetchone()[0]
    finally:
        con.close()
    assert n_sig == 1
    assert n_pos == 1


def test_e2e_mtm_closes_winning_position(e2e_db, fake_bars, monkeypatch):
    """After opening, simulate a bar that hits target → MTM should close."""
    from strategy_engine.live.detector import persist_signal, _detect_bollinger
    from strategy_engine.paper import (
        open_position_from_signal, list_positions, mark_to_market_all,
    )
    from dataclasses import asdict

    strat = _synthetic_bollinger_strategy()
    sig = _detect_bollinger(strat)
    persist_signal(sig)
    open_position_from_signal(asdict(sig))

    # Fake load_ohlcv to return a bar that's above the target
    target = sig.target_price
    winning_bar = pd.DataFrame({
        "open": [target + 1], "high": [target + 2], "low": [target - 0.5],
        "close": [target + 1], "volume": [100],
    }, index=[pd.Timestamp("2026-05-10")])

    def _fake_load(symbol, timeframe, start=None, end=None, **_):
        return winning_bar

    monkeypatch.setattr(
        "strategy_engine.paper.book.load_ohlcv", _fake_load, raising=False,
    )

    summary = mark_to_market_all()
    # At least one position should be closed now
    closed = list_positions(status="closed-target")
    assert len(closed) == 1
    assert closed[0]["strategy_id"] == "synth-spy-bol-v1"


def test_e2e_error_in_one_strategy_doesnt_stop_others(e2e_db, monkeypatch):
    """If one strategy detect raises, others still run AND the error is persisted."""
    from strategy_engine.live import detector as det

    class _S:
        def __init__(self, sid, tf="1d"): self.id = sid; self.timeframe = tf; self.status = "promoted"

    strats = [_S("ok-1"), _S("boom"), _S("ok-2")]
    monkeypatch.setattr(det, "validate_all", lambda: (strats, []))
    monkeypatch.setattr(det, "_is_due_now", lambda tf: True)

    calls = []
    def fake_detect(sid):
        calls.append(sid)
        if sid == "boom":
            raise RuntimeError("simulated crash")
        return None

    monkeypatch.setattr(det, "detect_signals_for_strategy", fake_detect)
    fired = det.detect_all_promoted(persist=False)
    assert calls == ["ok-1", "boom", "ok-2"]
    assert fired == []

    con = duckdb.connect(str(e2e_db), read_only=True)
    try:
        errs = con.execute("SELECT strategy_id FROM detect_errors").fetchall()
    finally:
        con.close()
    assert errs == [("boom",)]


def test_e2e_health_green_after_successful_cycle(e2e_db, fake_bars, monkeypatch, tmp_path):
    """After a normal detect + mtm cycle, health check should return OK."""
    from strategy_engine.live.detector import persist_signal, _detect_bollinger
    from strategy_engine.paper import open_position_from_signal
    from strategy_engine.live import health
    from dataclasses import asdict

    # Stub out launchd + other DB checks so health isolates to live-signals
    monkeypatch.setattr(health, "check_launchd_agents",
                          lambda loaded_fn=None: [health.Check("launchd", "ok", "stubbed")])
    monkeypatch.setattr(health, "check_databases",
                          lambda: [health.Check("db", "ok", "stubbed")])
    monkeypatch.setattr(health, "check_registry",
                          lambda: health.Check("reg", "ok", "stubbed"))
    monkeypatch.setattr(health, "check_promoted_dispatch",
                          lambda: [health.Check("disp", "ok", "stubbed")])

    # Run a full cycle
    strat = _synthetic_bollinger_strategy()
    sig = _detect_bollinger(strat)
    persist_signal(sig)
    open_position_from_signal(asdict(sig))

    report = health.run_health_check()
    assert report.status == "ok", [
        (c.name, c.severity, c.detail) for c in report.checks
    ]
