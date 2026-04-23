"""Behavioral tests for the EODHD intraday poller.

All tests use a tmp live-ticks DB and a fake EODHD client — no real network.
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from pathlib import Path
import duckdb
import pytest

from strategy_engine.live.intraday_poller import (
    PollTarget,
    collect_targets,
    poll_promoted,
    _upsert_bars,
)
from strategy_engine.registry.schema import Strategy


def _mk_strategy(
    *,
    status: str = "promoted",
    timeframe: str = "1h",
    symbol: str = "SPY",
    signal_type: str = "strat-pattern",
    strategy_id: str | None = None,
) -> Strategy:
    return Strategy.model_validate({
        "id": strategy_id or f"test-{symbol.lower()}-{timeframe}",
        "name": f"test {symbol} {timeframe}",
        "status": status,
        "asset_class": "equity-index",
        "instruments": [symbol],
        "timeframe": timeframe,
        "signal_logic": {"type": signal_type},
        "entry": {"mode": "trigger"},
        "exit": {"mode": "profit-target", "target": 0.05},
        "capital_allocation": 0.05,
        "data_sources": ["firstrate"],
    })


@pytest.fixture
def live_ticks_db(tmp_path, monkeypatch):
    db_path = tmp_path / "live-ticks.duckdb"
    # Re-point the module-level constant
    from strategy_engine.live import intraday_poller as poller
    monkeypatch.setattr(poller, "LIVE_TICKS_DB", db_path)

    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE ohlcv (
            symbol VARCHAR NOT NULL, datetime TIMESTAMP NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT,
            timeframe VARCHAR NOT NULL,
            source VARCHAR NOT NULL DEFAULT 'eodhd-intraday',
            fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, timeframe, datetime, source)
        )
    """)
    con.execute("""
        CREATE TABLE poll_log (
            poll_id VARCHAR PRIMARY KEY,
            started_at TIMESTAMP NOT NULL,
            finished_at TIMESTAMP,
            symbols VARCHAR, timeframes VARCHAR,
            bars_inserted INTEGER DEFAULT 0, bars_updated INTEGER DEFAULT 0,
            n_symbols_ok INTEGER DEFAULT 0, n_symbols_err INTEGER DEFAULT 0,
            error_summary VARCHAR, status VARCHAR NOT NULL DEFAULT 'running'
        )
    """)
    con.close()
    yield db_path


class FakeEODHD:
    """Stub EODHD client that returns pre-canned intraday bars per symbol."""

    def __init__(self, per_symbol_bars: dict[str, list[dict]] | None = None,
                 fail_symbols: set[str] | None = None):
        self.per_symbol_bars = per_symbol_bars or {}
        self.fail_symbols = fail_symbols or set()
        self.calls: list[tuple] = []

    def get_intraday(self, symbol: str, interval: str = "1h",
                     from_ts=None, to_ts=None):
        self.calls.append((symbol, interval, from_ts, to_ts))
        if symbol in self.fail_symbols:
            raise RuntimeError(f"EODHD simulated failure for {symbol}")
        return self.per_symbol_bars.get(symbol, [])


def _bar(ts: datetime, close: float = 500.0) -> dict:
    return {
        "timestamp": int(ts.timestamp()),
        "datetime": ts.isoformat(),
        "gmtoffset": 0,
        "open": close - 0.5, "high": close + 0.5, "low": close - 1.0,
        "close": close, "volume": 1_000_000,
    }


# ─── collect_targets ────────────────────────────────────────────────────────

def test_collect_targets_only_promoted_intraday():
    strategies = [
        _mk_strategy(status="promoted", timeframe="1h", symbol="SPY"),
        _mk_strategy(status="promoted", timeframe="4h", symbol="SPY"),  # same symbol+eodhd_interval
        _mk_strategy(status="promoted", timeframe="1d", symbol="QQQ"),  # daily — skipped
        _mk_strategy(status="draft", timeframe="1h", symbol="TSLA"),    # not promoted — skipped
        _mk_strategy(status="promoted", timeframe="1h", symbol="TSLA"),
    ]
    targets = collect_targets(strategies)
    symbols = sorted({(t.symbol, t.eodhd_interval) for t in targets})
    # SPY 1h + TSLA 1h only (1h and 4h dedupe to same eodhd interval)
    assert symbols == [("SPY", "1h"), ("TSLA", "1h")]


def test_collect_targets_empty_when_no_promoted():
    strategies = [_mk_strategy(status="draft", timeframe="1h", symbol="SPY")]
    assert collect_targets(strategies) == []


# ─── _upsert_bars ──────────────────────────────────────────────────────────

def test_upsert_bars_inserts_and_updates(live_ticks_db):
    from strategy_engine.live import intraday_poller as poller
    con = duckdb.connect(str(poller.LIVE_TICKS_DB))
    try:
        t0 = datetime(2026, 4, 23, 10, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 4, 23, 11, 0, tzinfo=timezone.utc)
        ins, upd = _upsert_bars(con, "SPY", "1hour", [_bar(t0, 500.0), _bar(t1, 501.0)])
        assert ins == 2
        assert upd == 0
        # Re-submit with updated close for t1
        ins2, upd2 = _upsert_bars(con, "SPY", "1hour",
                                  [_bar(t1, 502.5), _bar(t0 + timedelta(hours=2), 503.0)])
        assert upd2 == 1  # t1 re-upserted
        assert ins2 == 1  # new bar at 12:00 inserted
        # Final count: 3 rows
        n = con.execute("SELECT COUNT(*) FROM ohlcv WHERE symbol='SPY'").fetchone()[0]
        assert n == 3
        # t1's close is now 502.5
        close = con.execute(
            "SELECT close FROM ohlcv WHERE symbol='SPY' AND datetime = ?", [t1.replace(tzinfo=None)]
        ).fetchone()[0]
        assert close == 502.5
    finally:
        con.close()


# ─── poll_promoted end-to-end with fake client ─────────────────────────────

def test_poll_promoted_happy_path(live_ticks_db):
    strategies = [
        _mk_strategy(status="promoted", timeframe="1h", symbol="SPY"),
        _mk_strategy(status="promoted", timeframe="1h", symbol="TSLA"),
    ]
    t0 = datetime(2026, 4, 23, 10, 0, tzinfo=timezone.utc)
    client = FakeEODHD(per_symbol_bars={
        "SPY": [_bar(t0, 500.0), _bar(t0 + timedelta(hours=1), 501.0)],
        "TSLA": [_bar(t0, 250.0)],
    })
    result = poll_promoted(strategies=strategies, client=client)
    assert result.status == "ok"
    assert result.n_ok == 2
    assert result.n_err == 0
    assert result.bars_inserted == 3
    # Client was called with matching arguments per symbol
    assert {c[0] for c in client.calls} == {"SPY", "TSLA"}
    assert all(c[1] == "1h" for c in client.calls)


def test_poll_promoted_partial_failure(live_ticks_db):
    strategies = [
        _mk_strategy(status="promoted", timeframe="1h", symbol="SPY"),
        _mk_strategy(status="promoted", timeframe="1h", symbol="FAIL"),
    ]
    t0 = datetime(2026, 4, 23, 10, 0, tzinfo=timezone.utc)
    client = FakeEODHD(
        per_symbol_bars={"SPY": [_bar(t0, 500.0)]},
        fail_symbols={"FAIL"},
    )
    result = poll_promoted(strategies=strategies, client=client)
    assert result.status == "partial"
    assert result.n_ok == 1
    assert result.n_err == 1
    assert result.bars_inserted == 1
    assert any("FAIL" in e for e in result.errors)


def test_poll_promoted_empty_registry(live_ticks_db):
    result = poll_promoted(strategies=[], client=FakeEODHD())
    assert result.status == "ok-empty"
    assert result.bars_inserted == 0


def test_poll_log_row_written(live_ticks_db):
    strategies = [_mk_strategy(status="promoted", timeframe="1h", symbol="SPY")]
    t0 = datetime(2026, 4, 23, 10, 0, tzinfo=timezone.utc)
    client = FakeEODHD(per_symbol_bars={"SPY": [_bar(t0, 500.0)]})
    result = poll_promoted(strategies=strategies, client=client)

    from strategy_engine.live import intraday_poller as poller
    con = duckdb.connect(str(poller.LIVE_TICKS_DB))
    try:
        row = con.execute(
            "SELECT poll_id, status, bars_inserted, n_symbols_ok FROM poll_log "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    finally:
        con.close()
    assert row[0] == result.poll_id
    assert row[1] == "ok"
    assert row[2] == 1
    assert row[3] == 1
