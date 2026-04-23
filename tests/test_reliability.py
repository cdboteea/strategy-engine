"""Behavioral tests for detector error isolation + notification retry."""
from __future__ import annotations
from pathlib import Path
from unittest.mock import patch
import duckdb
import pytest


# ── Fixtures ────────────────────────────────────────────────────────────────

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

    from strategy_engine.live import detector, notification
    monkeypatch.setattr(detector, "LIVE_DB", db)
    monkeypatch.setattr(notification, "LIVE_DB", db)
    yield db


# ── A2: detector error isolation ────────────────────────────────────────────

def test_detect_error_is_persisted(live_db):
    """A failing strategy detect should write a detect_errors row (not crash batch)."""
    from strategy_engine.live.detector import _log_detect_error

    try:
        raise ValueError("synthetic failure")
    except ValueError as e:
        eid = _log_detect_error("run-test", "some-strategy", e)

    assert eid.startswith("err-")
    con = duckdb.connect(str(live_db), read_only=True)
    try:
        rows = con.execute(
            "SELECT strategy_id, error_type, error_message, run_id FROM detect_errors"
        ).fetchall()
    finally:
        con.close()
    assert len(rows) == 1
    assert rows[0] == ("some-strategy", "ValueError", "synthetic failure", "run-test")


def test_detect_all_promoted_isolates_per_strategy_failures(live_db, monkeypatch):
    """One strategy raising should NOT stop other strategies from running."""
    from strategy_engine.live import detector as det

    # Fake strategy objects
    class _FakeStrat:
        def __init__(self, sid, tf="1w"): self.id = sid; self.timeframe = tf; self.status = "promoted"

    fakes = [_FakeStrat("ok-1"), _FakeStrat("boom"), _FakeStrat("ok-2")]
    monkeypatch.setattr(det, "validate_all", lambda: (fakes, []))
    monkeypatch.setattr(det, "_is_due_now", lambda tf: True)

    call_log = []
    def fake_detect(sid):
        call_log.append(sid)
        if sid == "boom":
            raise RuntimeError("intentional explosion")
        return None  # no signal

    monkeypatch.setattr(det, "detect_signals_for_strategy", fake_detect)
    fired = det.detect_all_promoted(persist=False)

    # All 3 got called despite one blowing up
    assert call_log == ["ok-1", "boom", "ok-2"]
    assert fired == []
    # Error persisted for 'boom'
    con = duckdb.connect(str(live_db), read_only=True)
    rows = con.execute("SELECT strategy_id FROM detect_errors").fetchall()
    con.close()
    assert rows == [("boom",)]


def test_recent_detect_errors_returns_fresh_entries(live_db):
    from strategy_engine.live.detector import _log_detect_error, recent_detect_errors

    for sid in ["a", "b", "c"]:
        try: raise RuntimeError(f"err-{sid}")
        except RuntimeError as e: _log_detect_error("run-x", sid, e)

    rows = recent_detect_errors(hours=1)
    assert len(rows) == 3
    # Most recent first (ORDER BY error_at DESC)
    assert [r["strategy_id"] for r in rows] == ["c", "b", "a"]


# ── A1: Telegram retry + notification logging ──────────────────────────────

def test_telegram_succeeds_on_first_attempt(live_db, monkeypatch):
    from strategy_engine.live import notification as n

    calls = []
    monkeypatch.setattr(n, "_send_telegram_once", lambda msg, timeout=3.0: calls.append(msg))
    ok, attempts, err = n.send_telegram_with_retry("hello", sleep_fn=lambda s: None)

    assert ok is True
    assert attempts == 1
    assert err == ""
    assert len(calls) == 1


def test_telegram_retries_on_transient_failure(live_db, monkeypatch):
    from strategy_engine.live import notification as n

    attempts_made = [0]
    def flaky(msg, timeout=3.0):
        attempts_made[0] += 1
        if attempts_made[0] < 3:
            raise ConnectionError("gateway down")
        return None

    monkeypatch.setattr(n, "_send_telegram_once", flaky)
    ok, attempts, err = n.send_telegram_with_retry("hi", sleep_fn=lambda s: None)

    assert ok is True
    assert attempts == 3
    assert err == ""


def test_telegram_gives_up_after_max_retries(live_db, monkeypatch):
    from strategy_engine.live import notification as n

    def always_fail(msg, timeout=3.0):
        raise TimeoutError("stuck")

    monkeypatch.setattr(n, "_send_telegram_once", always_fail)
    ok, attempts, err = n.send_telegram_with_retry("hi", max_retries=3, sleep_fn=lambda s: None)

    assert ok is False
    assert attempts == 3
    assert "TimeoutError" in err


def test_notify_signal_falls_back_to_stdout_and_logs(live_db, monkeypatch, capsys):
    from strategy_engine.live import notification as n
    from strategy_engine.live.detector import SignalFired

    monkeypatch.setattr(n, "send_telegram_with_retry",
                        lambda msg, **kw: (False, 3, "ConnectionError: down"))

    sig = SignalFired(
        signal_id="sig-test", strategy_id="xyz",
        fired_at="2026-04-23T10:00:00-04:00", bar_timestamp="2026-04-22",
        symbol="SPY", timeframe="1w", signal_type="bollinger-lower-band",
        pattern=None, direction="bullish", ftfc_aligned=None,
        entry_price=500.0, stop_price=None, target_price=525.0,
        recommended_size=0.10, metadata="{}",
    )
    used = n.notify_signal(sig, channel="auto")
    assert used == "stdout"
    # Both telegram failure AND stdout success should be logged
    con = duckdb.connect(str(live_db), read_only=True)
    rows = con.execute(
        "SELECT channel, status, attempts FROM notification_log ORDER BY attempted_at"
    ).fetchall()
    con.close()
    assert rows == [("telegram", "fell-back-stdout", 3), ("stdout", "sent", 1)]


def test_notify_signal_logs_telegram_success(live_db, monkeypatch):
    from strategy_engine.live import notification as n
    from strategy_engine.live.detector import SignalFired

    monkeypatch.setattr(n, "send_telegram_with_retry", lambda msg, **kw: (True, 1, ""))

    sig = SignalFired(
        signal_id="sig-ok", strategy_id="xyz",
        fired_at="2026-04-23T10:00:00-04:00", bar_timestamp="2026-04-22",
        symbol="SPY", timeframe="1w", signal_type="bollinger-lower-band",
        pattern=None, direction="bullish", ftfc_aligned=None,
        entry_price=500.0, stop_price=None, target_price=525.0,
        recommended_size=0.10, metadata="{}",
    )
    used = n.notify_signal(sig, channel="auto")
    assert used == "telegram"
    con = duckdb.connect(str(live_db), read_only=True)
    rows = con.execute(
        "SELECT channel, status FROM notification_log"
    ).fetchall()
    con.close()
    assert rows == [("telegram", "sent")]


def test_check_gateway_health_handles_unreachable():
    from strategy_engine.live.notification import check_gateway_health
    from unittest.mock import patch
    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("nope")):
        ok, detail = check_gateway_health(timeout=0.1)
    assert ok is False
    assert "URLError" in detail or "unreachable" in detail
