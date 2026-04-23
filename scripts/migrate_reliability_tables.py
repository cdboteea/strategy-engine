"""
Add reliability tables to live-signals.duckdb:

  detect_errors       — one row per per-strategy detect failure (traceback)
  notification_log    — one row per notification attempt (retries + status)

Both feed the Stage-A error-isolation + telegram-retry work. Safe to run
multiple times (CREATE TABLE IF NOT EXISTS).
"""
from __future__ import annotations
from pathlib import Path
import duckdb


LIVE_DB = Path.home() / "clawd" / "data" / "live-signals.duckdb"


def main() -> int:
    con = duckdb.connect(str(LIVE_DB))
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS detect_errors (
                error_id       VARCHAR PRIMARY KEY,
                run_id         VARCHAR,                   -- batch UUID (all errors from one cron run share this)
                strategy_id    VARCHAR NOT NULL,
                error_at       TIMESTAMP NOT NULL,
                error_type     VARCHAR NOT NULL,          -- e.g. 'DataNotAvailable', 'ValueError'
                error_message  VARCHAR NOT NULL,
                traceback_text VARCHAR,
                engine_version VARCHAR
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_detect_errors_strategy ON detect_errors (strategy_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_detect_errors_at ON detect_errors (error_at DESC)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_detect_errors_run ON detect_errors (run_id)")

        con.execute("""
            CREATE TABLE IF NOT EXISTS notification_log (
                notif_id      VARCHAR PRIMARY KEY,
                signal_id     VARCHAR,                   -- FK live_signals.signal_id (nullable for test pings)
                channel       VARCHAR NOT NULL,          -- 'telegram' | 'stdout'
                attempted_at  TIMESTAMP NOT NULL,
                attempts      INTEGER NOT NULL DEFAULT 1,
                status        VARCHAR NOT NULL,          -- 'sent' | 'failed' | 'fell-back-stdout'
                error_message VARCHAR,
                elapsed_ms    INTEGER
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_notif_signal ON notification_log (signal_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_notif_at ON notification_log (attempted_at DESC)")

        print("✓ detect_errors + notification_log tables ready")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
