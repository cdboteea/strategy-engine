"""
Initialize paper-trading tables inside live-signals.duckdb.

Tables:
  paper_positions       — open + closed positions opened from detected signals
  paper_nav_snapshots   — daily NAV of the paper book (for equity curve)

Schema:
  paper_positions (
    position_id           VARCHAR PRIMARY KEY   — pp-<signal_id> (1:1 with signal)
    signal_id             VARCHAR REFERENCES live_signals  — which signal opened it
    strategy_id           VARCHAR
    symbol                VARCHAR
    timeframe             VARCHAR
    direction             VARCHAR
    opened_at             TIMESTAMP
    opened_price          DOUBLE                 — filled-at price (signal's entry_trigger for STRAT; signal's close for Bollinger)
    target_price          DOUBLE
    stop_price            DOUBLE
    size_fraction         DOUBLE                 — capital_allocation
    notional_size         DOUBLE                 — opening NAV × size_fraction
    status                VARCHAR                — 'open' | 'closed-target' | 'closed-stop' | 'closed-window' | 'closed-manual'
    closed_at             TIMESTAMP
    closed_price          DOUBLE
    realized_pct_return   DOUBLE                 — (closed - opened)/opened, sign-adjusted for bearish
    realized_pnl_usd      DOUBLE                 — notional × realized_pct_return
    holding_bars          INTEGER
    last_mtm_at           TIMESTAMP
    last_mtm_price        DOUBLE
    unrealized_pct_return DOUBLE                 — updated daily until close
    metadata              VARCHAR                — JSON

  paper_nav_snapshots (
    snap_date   DATE PRIMARY KEY
    nav_usd     DOUBLE         — total paper book NAV (starts at $100,000)
    n_open      INTEGER
    n_closed_today  INTEGER
    realized_today  DOUBLE
    unrealized      DOUBLE
  )

Initial book NAV is $100,000. Each position sizes at capital_allocation × current_nav.
"""
from __future__ import annotations
from pathlib import Path
import duckdb


LIVE_DB = Path.home() / "clawd" / "data" / "live-signals.duckdb"
INITIAL_NAV = 100_000.0


def main() -> int:
    con = duckdb.connect(str(LIVE_DB))
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS paper_positions (
                position_id           VARCHAR PRIMARY KEY,
                signal_id             VARCHAR NOT NULL,
                strategy_id           VARCHAR NOT NULL,
                symbol                VARCHAR NOT NULL,
                timeframe             VARCHAR NOT NULL,
                direction             VARCHAR NOT NULL,
                opened_at             TIMESTAMP NOT NULL,
                opened_price          DOUBLE NOT NULL,
                target_price          DOUBLE,
                stop_price            DOUBLE,
                size_fraction         DOUBLE NOT NULL,
                notional_size         DOUBLE NOT NULL,
                status                VARCHAR DEFAULT 'open',
                closed_at             TIMESTAMP,
                closed_price          DOUBLE,
                realized_pct_return   DOUBLE,
                realized_pnl_usd      DOUBLE,
                holding_bars          INTEGER,
                last_mtm_at           TIMESTAMP,
                last_mtm_price        DOUBLE,
                unrealized_pct_return DOUBLE,
                metadata              VARCHAR
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_pp_status ON paper_positions (status)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_pp_strategy ON paper_positions (strategy_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_pp_opened ON paper_positions (opened_at DESC)")

        con.execute("""
            CREATE TABLE IF NOT EXISTS paper_nav_snapshots (
                snap_date         DATE PRIMARY KEY,
                nav_usd           DOUBLE NOT NULL,
                n_open            INTEGER,
                n_closed_today    INTEGER,
                realized_today    DOUBLE,
                unrealized        DOUBLE
            )
        """)

        # Seed initial NAV snapshot if empty
        count = con.execute("SELECT COUNT(*) FROM paper_nav_snapshots").fetchone()[0]
        if count == 0:
            con.execute(
                "INSERT INTO paper_nav_snapshots (snap_date, nav_usd, n_open, n_closed_today, realized_today, unrealized) "
                "VALUES (current_date, ?, 0, 0, 0.0, 0.0)",
                [INITIAL_NAV],
            )
            print(f"  Seeded initial NAV: ${INITIAL_NAV:,.2f}")

        n_pos = con.execute("SELECT COUNT(*) FROM paper_positions").fetchone()[0]
        n_snap = con.execute("SELECT COUNT(*) FROM paper_nav_snapshots").fetchone()[0]
        print(f"✓ paper_positions ready   (rows: {n_pos})")
        print(f"✓ paper_nav_snapshots ready  (rows: {n_snap})")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
