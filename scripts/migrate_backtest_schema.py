"""
Schema migration — backtest-results.duckdb.

v0 → v1:
- Add `retired_at` TIMESTAMP (nullable) — when a run was retired
- Add `retirement_reason` VARCHAR (nullable)
- Add `engine_version` VARCHAR (nullable) — strategy-engine version that wrote the row
- Add `registry_yaml_path` VARCHAR (nullable) — link back to source YAML
- Annotate the 15 crypto rows from 2026-03-11 as retired per execution-plan-final-2026-04-22

Idempotent — checks for existing columns before ALTER TABLE.

Run:
    python scripts/migrate_backtest_schema.py
"""
from __future__ import annotations
from datetime import datetime
import duckdb

from strategy_engine.config import BACKTEST_DB


RETIREMENT_REASON = (
    "Formal retirement 2026-04-22 per execution-plan-final-2026-04-22. "
    "All 15 runs from 2026-03-11 on crypto had OOS Sharpe <= 0.15; "
    "paper-trading engine retired 2026-04-19; crypto track paused until "
    "SPY-Bollinger + STRAT are live in traditional markets, then revisit on "
    "Hyperliquid."
)


def existing_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    rows = con.execute(f"DESCRIBE {table}").fetchall()
    return {r[0] for r in rows}


def ensure_column(con, table: str, col: str, coltype: str) -> bool:
    if col in existing_columns(con, table):
        return False
    con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
    print(f"  added column: {col} {coltype}")
    return True


def main() -> int:
    print(f"Opening {BACKTEST_DB}")
    con = duckdb.connect(str(BACKTEST_DB))

    # 1. Schema extensions
    print("\n→ Extending schema")
    ensure_column(con, "backtest_results", "retired_at", "TIMESTAMP")
    ensure_column(con, "backtest_results", "retirement_reason", "VARCHAR")
    ensure_column(con, "backtest_results", "engine_version", "VARCHAR")
    ensure_column(con, "backtest_results", "registry_yaml_path", "VARCHAR")

    # 2. Retire the 15 crypto rows from 2026-03-11
    print("\n→ Retiring 15 crypto runs from 2026-03-11")
    before = con.execute(
        "SELECT COUNT(*) FROM backtest_results WHERE run_date::date = '2026-03-11' AND retired_at IS NULL"
    ).fetchone()[0]
    print(f"  found {before} candidate row(s)")

    now = datetime.now().isoformat(timespec="seconds")
    con.execute(
        """
        UPDATE backtest_results
        SET retired_at = ?, retirement_reason = ?
        WHERE run_date::date = '2026-03-11' AND retired_at IS NULL
        """,
        [now, RETIREMENT_REASON],
    )
    after = con.execute(
        "SELECT COUNT(*) FROM backtest_results WHERE retired_at IS NOT NULL"
    ).fetchone()[0]
    print(f"  total retired: {after}")

    # 3. Verify
    print("\n→ Verification")
    cols = existing_columns(con, "backtest_results")
    print(f"  columns: {sorted(cols)}")
    active = con.execute(
        "SELECT COUNT(*) FROM backtest_results WHERE retired_at IS NULL"
    ).fetchone()[0]
    retired = con.execute(
        "SELECT COUNT(*) FROM backtest_results WHERE retired_at IS NOT NULL"
    ).fetchone()[0]
    print(f"  active runs: {active}")
    print(f"  retired runs: {retired}")

    con.close()
    print("\n✓ Migration complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
