"""
System health check for the strategy engine.

Produces a single pass/warn/fail report covering:
  1. Every DB file opens cleanly + expected tables exist
  2. Every launchd plist we care about is loaded
  3. Recent detect-error density (warn if > threshold in last 24h)
  4. Recent notification failure rate (warn if Telegram > 50% failed)
  5. Paper book invariants (no orphan positions, NAV > 0)
  6. Registry validates cleanly
  7. Every promoted strategy can be loaded and its signal logic dispatched

Severity levels:
  ok      — all checks green
  warn    — non-critical issue (e.g. stale intraday data, gateway offline)
  error   — critical (DB unreadable, registry invalid, etc.)

Exit codes when called from CLI: 0 ok, 1 warnings, 2 errors.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import json
import subprocess

import duckdb

from ..config import BACKTEST_DB, LIVE_TICKS_DB
from ..registry.loader import validate_all


LIVE_DB = Path.home() / "clawd" / "data" / "live-signals.duckdb"

# Expected launchd agents we installed. A missing one is a warning (user may
# have intentionally unloaded it); an agent listed but repeatedly failing
# would show up in logs, not here.
EXPECTED_AGENTS = [
    "com.matias.strategy-engine.weekly-detect",
    "com.matias.strategy-engine.daily-detect",
    "com.matias.strategy-engine.paper-mtm",
    "com.matias.strategy-engine.intraday-poll",
    "com.matias.ideas-auto-archive",
    "com.matias.ideas-gmail-tostage",
    "com.matias.ideas-x-bookmarks",
    "com.matias.ideas-pdf-dropfolder",
    "com.matias.ideas-telegram-queue",
    "com.matias.ideas-graphiti-ingest",
    "com.matias.rotate-logs",
]

# Tables each DB should have. If a table is missing, a migration is probably
# needed; we flag as error.
EXPECTED_TABLES = {
    "backtest-results.duckdb": ["backtest_results"],
    "live-signals.duckdb": ["live_signals", "paper_positions", "paper_nav_snapshots",
                             "detect_errors", "notification_log"],
    "live-ticks.duckdb": ["ohlcv", "poll_log"],
}

DETECT_ERROR_WARN_THRESHOLD = 5          # errors in last 24h → warn
TELEGRAM_FAIL_RATE_WARN = 0.5            # >50% telegram failures → warn


Severity = Literal["ok", "warn", "error"]


@dataclass
class Check:
    name: str
    severity: Severity
    detail: str
    extras: dict = field(default_factory=dict)


@dataclass
class HealthReport:
    checked_at: str
    status: Severity           # worst severity across all checks
    checks: list[Check] = field(default_factory=list)

    def summary_line(self) -> str:
        n_ok = sum(1 for c in self.checks if c.severity == "ok")
        n_warn = sum(1 for c in self.checks if c.severity == "warn")
        n_err = sum(1 for c in self.checks if c.severity == "error")
        return f"{self.status.upper()} — {n_ok} ok, {n_warn} warn, {n_err} error"


def _check_db(db_path: Path, expected_tables: list[str]) -> Check:
    name = f"db:{db_path.name}"
    if not db_path.exists():
        return Check(name, "error", f"missing: {db_path}")
    try:
        con = duckdb.connect(str(db_path), read_only=True)
    except duckdb.IOException as e:
        return Check(name, "warn", f"locked: {e}")
    except Exception as e:
        return Check(name, "error", f"open failed: {e}")
    try:
        existing = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    finally:
        con.close()
    missing = [t for t in expected_tables if t not in existing]
    if missing:
        return Check(name, "error", f"missing tables: {missing}", {"existing": sorted(existing)})
    return Check(name, "ok", f"{len(existing)} tables present")


def check_databases() -> list[Check]:
    out: list[Check] = []
    for db_fname, tables in EXPECTED_TABLES.items():
        if db_fname == "backtest-results.duckdb":
            out.append(_check_db(BACKTEST_DB, tables))
        elif db_fname == "live-signals.duckdb":
            out.append(_check_db(LIVE_DB, tables))
        elif db_fname == "live-ticks.duckdb":
            out.append(_check_db(LIVE_TICKS_DB, tables))
    return out


def _loaded_agents() -> set[str]:
    """Return set of matias-scope agent labels currently loaded via launchd."""
    try:
        out = subprocess.run(
            ["/bin/launchctl", "list"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return set()
    labels: set[str] = set()
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[2].startswith("com.matias."):
            labels.add(parts[2])
    return labels


def check_launchd_agents(loaded_fn=_loaded_agents) -> list[Check]:
    loaded = loaded_fn()
    if not loaded:
        return [Check("launchd", "warn",
                      "no com.matias.* agents loaded (is launchd running?)")]
    out: list[Check] = []
    for label in EXPECTED_AGENTS:
        if label in loaded:
            out.append(Check(f"launchd:{label}", "ok", "loaded"))
        else:
            out.append(Check(f"launchd:{label}", "warn", "not loaded"))
    return out


def check_recent_detect_errors(hours: int = 24,
                                 threshold: int = DETECT_ERROR_WARN_THRESHOLD) -> Check:
    if not LIVE_DB.exists():
        return Check("detect-errors", "warn", "live-signals.duckdb missing")
    try:
        con = duckdb.connect(str(LIVE_DB), read_only=True)
    except duckdb.IOException:
        return Check("detect-errors", "warn", "live-signals.duckdb locked")
    try:
        row = con.execute(
            "SELECT COUNT(*) FROM detect_errors "
            "WHERE error_at >= CURRENT_TIMESTAMP - INTERVAL (?) HOUR",
            [hours],
        ).fetchone()
    finally:
        con.close()
    n = row[0] if row else 0
    if n == 0:
        return Check("detect-errors", "ok", f"0 errors in last {hours}h")
    if n <= threshold:
        return Check("detect-errors", "ok",
                      f"{n} error(s) in last {hours}h (under threshold {threshold})")
    return Check("detect-errors", "warn",
                  f"{n} error(s) in last {hours}h (over threshold {threshold})")


def check_notification_health(hours: int = 24,
                                 fail_rate_threshold: float = TELEGRAM_FAIL_RATE_WARN) -> Check:
    if not LIVE_DB.exists():
        return Check("notifications", "warn", "live-signals.duckdb missing")
    try:
        con = duckdb.connect(str(LIVE_DB), read_only=True)
    except duckdb.IOException:
        return Check("notifications", "warn", "live-signals.duckdb locked")
    try:
        row = con.execute(
            """
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN channel = 'telegram' AND status = 'sent' THEN 1 ELSE 0 END) AS tg_ok,
              SUM(CASE WHEN channel = 'telegram' AND status != 'sent' THEN 1 ELSE 0 END) AS tg_fail
            FROM notification_log
            WHERE attempted_at >= CURRENT_TIMESTAMP - INTERVAL (?) HOUR
            """,
            [hours],
        ).fetchone()
    finally:
        con.close()
    total, tg_ok, tg_fail = (row or (0, 0, 0))
    total = total or 0
    tg_ok = tg_ok or 0
    tg_fail = tg_fail or 0
    if total == 0:
        return Check("notifications", "ok", f"0 attempts in last {hours}h")
    tg_total = tg_ok + tg_fail
    if tg_total == 0:
        return Check("notifications", "ok", f"{total} attempts; no Telegram traffic")
    fail_rate = tg_fail / tg_total
    if fail_rate >= fail_rate_threshold:
        return Check("notifications", "warn",
                      f"{tg_fail}/{tg_total} Telegram failed ({fail_rate:.0%})",
                      {"total": total, "tg_ok": tg_ok, "tg_fail": tg_fail})
    return Check("notifications", "ok",
                  f"{tg_ok}/{tg_total} Telegram sent ({(1-fail_rate):.0%})")


def check_paper_book_invariants() -> Check:
    if not LIVE_DB.exists():
        return Check("paper-book", "warn", "live-signals.duckdb missing")
    try:
        con = duckdb.connect(str(LIVE_DB), read_only=True)
    except duckdb.IOException:
        return Check("paper-book", "warn", "live-signals.duckdb locked")
    try:
        nav = con.execute(
            "SELECT nav_usd FROM paper_nav_snapshots ORDER BY snap_date DESC LIMIT 1"
        ).fetchone()
        # Orphan: position in status 'open' whose opened_at is older than 13 weeks
        # (longer than any forward window — means MTM missed closing)
        stale_open = con.execute(
            "SELECT COUNT(*) FROM paper_positions "
            "WHERE status = 'open' AND opened_at < CURRENT_TIMESTAMP - INTERVAL 14 WEEK"
        ).fetchone()[0]
    finally:
        con.close()
    nav_val = float(nav[0]) if nav else None
    if nav_val is None:
        return Check("paper-book", "warn", "no NAV snapshot yet")
    if nav_val <= 0:
        return Check("paper-book", "error", f"NAV ≤ 0 ({nav_val})")
    if stale_open > 0:
        return Check("paper-book", "warn",
                      f"{stale_open} open position(s) past 14-week MTM window",
                      {"nav": nav_val, "stale_open": stale_open})
    return Check("paper-book", "ok", f"NAV ${nav_val:,.2f}, no orphans")


def check_registry() -> Check:
    try:
        ok_list, errors = validate_all()
    except Exception as e:
        return Check("registry", "error", f"validate_all raised: {e}")
    if errors:
        return Check("registry", "error",
                      f"{len(errors)} validation error(s)",
                      {"first_3": [str(e[1])[:120] for e in errors[:3]]})
    return Check("registry", "ok", f"{len(ok_list)} strategies validated")


def check_promoted_dispatch(max_strategies: int = 20) -> list[Check]:
    """Ensure every promoted strategy's signal_logic.type is dispatchable.

    We don't run the whole detect path (expensive + needs live data) — we
    just verify each strategy's type is one the runner knows about.
    """
    try:
        ok_list, _ = validate_all()
    except Exception as e:
        return [Check("promoted-dispatch", "error", f"can't list strategies: {e}")]
    promoted = [s for s in ok_list if s.status == "promoted"][:max_strategies]
    supported = {
        "bollinger-mean-reversion", "strat-pattern", "composite",
        "sma-crossover", "macd-crossover", "donchian-breakout", "trend-pullback",
    }
    out: list[Check] = []
    n_ok = 0
    for s in promoted:
        t = s.signal_logic.type
        if t in supported:
            n_ok += 1
        else:
            out.append(Check(f"dispatch:{s.id}", "error", f"unsupported type {t!r}"))
    out.insert(0, Check("promoted-dispatch", "ok" if not out else "error",
                         f"{n_ok}/{len(promoted)} promoted strategies dispatchable"))
    return out


def run_health_check() -> HealthReport:
    checks: list[Check] = []
    checks.extend(check_databases())
    checks.extend(check_launchd_agents())
    checks.append(check_recent_detect_errors())
    checks.append(check_notification_health())
    checks.append(check_paper_book_invariants())
    checks.append(check_registry())
    checks.extend(check_promoted_dispatch())

    # Aggregate severity
    severity: Severity = "ok"
    for c in checks:
        if c.severity == "error":
            severity = "error"
            break
        elif c.severity == "warn" and severity == "ok":
            severity = "warn"

    return HealthReport(
        checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        status=severity,
        checks=checks,
    )
