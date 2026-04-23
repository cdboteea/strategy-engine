"""
Initialize ~/clawd/data/live-signals.duckdb schema.

Table: live_signals — one row per real-time signal fire.

Columns:
  signal_id              TEXT PRIMARY KEY  — uuid-ish, e.g. sig-<strategy>-<timestamp>
  strategy_id            TEXT              — FK to registry YAML id
  fired_at               TIMESTAMP         — when the signal detector fired
  bar_timestamp          TIMESTAMP         — the bar that triggered
  symbol                 TEXT
  timeframe              TEXT
  signal_type            TEXT              — 'bollinger-lower-band' | 'strat-pattern' | ...
  pattern                TEXT NULL         — for STRAT: '2d-2u' etc.
  direction              TEXT              — 'bullish' | 'bearish' | 'neutral'
  ftfc_aligned           BOOLEAN NULL      — for STRAT: was FTFC satisfied
  entry_price            DOUBLE NULL       — if the strategy specifies one
  stop_price             DOUBLE NULL
  target_price           DOUBLE NULL
  recommended_size       DOUBLE NULL       — from capital_allocation + portfolio state
  notification_sent      BOOLEAN
  notification_channel   TEXT NULL         — 'telegram' | 'cli' | ...
  status                 TEXT              — 'new' | 'acknowledged' | 'executed' | 'expired' | 'ignored'
  executed_at            TIMESTAMP NULL
  executed_price         DOUBLE NULL
  exit_at                TIMESTAMP NULL
  exit_price             DOUBLE NULL
  exit_reason            TEXT NULL         — 'target' | 'stop' | 'manual' | 'expired'
  realized_return        DOUBLE NULL
  engine_version         TEXT
  metadata               TEXT              — JSON payload for strategy-specific extras

Indexes:
  strategy_id, status, fired_at DESC
"""
from __future__ import annotations
from pathlib import Path
import duckdb


LIVE_DB = Path.home() / "clawd" / "data" / "live-signals.duckdb"


def main() -> int:
    LIVE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(LIVE_DB))
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS live_signals (
                signal_id             VARCHAR PRIMARY KEY,
                strategy_id           VARCHAR NOT NULL,
                fired_at              TIMESTAMP NOT NULL,
                bar_timestamp         TIMESTAMP NOT NULL,
                symbol                VARCHAR NOT NULL,
                timeframe             VARCHAR NOT NULL,
                signal_type           VARCHAR NOT NULL,
                pattern               VARCHAR,
                direction             VARCHAR,
                ftfc_aligned          BOOLEAN,
                entry_price           DOUBLE,
                stop_price            DOUBLE,
                target_price          DOUBLE,
                recommended_size      DOUBLE,
                notification_sent     BOOLEAN DEFAULT FALSE,
                notification_channel  VARCHAR,
                status                VARCHAR DEFAULT 'new',
                executed_at           TIMESTAMP,
                executed_price        DOUBLE,
                exit_at               TIMESTAMP,
                exit_price            DOUBLE,
                exit_reason           VARCHAR,
                realized_return       DOUBLE,
                engine_version        VARCHAR,
                metadata              VARCHAR
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_signals_strategy ON live_signals (strategy_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON live_signals (status)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_signals_fired ON live_signals (fired_at DESC)")

        count = con.execute("SELECT COUNT(*) FROM live_signals").fetchone()[0]
        print(f"✓ live_signals table ready at {LIVE_DB}")
        print(f"  current row count: {count}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
