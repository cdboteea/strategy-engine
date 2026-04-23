"""
Initialize ~/clawd/data/live-ticks.duckdb — the fresh intraday splice DB.

Stores recent intraday bars fetched from EODHD intraday API. The
duckdb provider splices this into queries for {1m, 5m, 15m, 30m, 1h}
timeframes so live detectors can operate on current data rather than
stopping at firstrate's monthly refresh ceiling.

Schema is deliberately close to firstrate.ohlcv so the splice is a
simple UNION ALL with a timeframe-name rewrite.
"""
from __future__ import annotations
from pathlib import Path
import duckdb


LIVE_TICKS_DB = Path.home() / "clawd" / "data" / "live-ticks.duckdb"


def main() -> None:
    con = duckdb.connect(str(LIVE_TICKS_DB))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol      VARCHAR NOT NULL,
                datetime    TIMESTAMP NOT NULL,
                open        DOUBLE,
                high        DOUBLE,
                low         DOUBLE,
                close       DOUBLE,
                volume      BIGINT,
                timeframe   VARCHAR NOT NULL,
                source      VARCHAR NOT NULL DEFAULT 'eodhd-intraday',
                fetched_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timeframe, datetime, source)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS poll_log (
                poll_id       VARCHAR PRIMARY KEY,
                started_at    TIMESTAMP NOT NULL,
                finished_at   TIMESTAMP,
                symbols       VARCHAR,
                timeframes    VARCHAR,
                bars_inserted INTEGER DEFAULT 0,
                bars_updated  INTEGER DEFAULT 0,
                n_symbols_ok  INTEGER DEFAULT 0,
                n_symbols_err INTEGER DEFAULT 0,
                error_summary VARCHAR,
                status        VARCHAR NOT NULL DEFAULT 'running'
            )
            """
        )
        print("OK — live-ticks.duckdb initialized at", LIVE_TICKS_DB)
    finally:
        con.close()


if __name__ == "__main__":
    main()
