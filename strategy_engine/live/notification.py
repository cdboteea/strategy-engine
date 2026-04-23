"""
Signal notification channels.

Primary: Telegram via OpenClaw gateway (port 18789).
Fallback: stdout.

Every attempt (success or failure) is logged to live-signals.notification_log
with timing, retry count, and error detail. This gives us a paper trail when
a signal fires but the user never sees it.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import json
import subprocess
import time
import urllib.request
import urllib.error
import uuid

import duckdb

from .detector import SignalFired


OPENCLAW_GATEWAY = "http://127.0.0.1:18789"
OPENCLAW_TIMEOUT = 3.0
MAX_RETRIES = 3
BASE_BACKOFF_S = 0.5   # grows to 0.5s, 1.0s, 2.0s

LIVE_DB = Path.home() / "clawd" / "data" / "live-signals.duckdb"


# ── Health check ────────────────────────────────────────────────────────────

def check_gateway_health(timeout: float = 1.0) -> tuple[bool, str]:
    """
    Ping the OpenClaw gateway. Returns (healthy, detail).

    We try `/health` first; if that 404s, we fall back to a HEAD on root.
    """
    try:
        req = urllib.request.Request(f"{OPENCLAW_GATEWAY}/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status < 400:
                return True, f"gateway OK ({resp.status})"
            return False, f"gateway HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        # 404 on /health is fine if gateway responds at all — treat as reachable
        if e.code == 404:
            return True, "gateway reachable (no /health endpoint)"
        return False, f"HTTPError {e.code}"
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return False, f"unreachable: {e.__class__.__name__}"


# ── Message formatting ──────────────────────────────────────────────────────

def format_signal(sig: SignalFired) -> str:
    """Human-readable summary of a fired signal."""
    lines = [
        f"🔔 *{sig.strategy_id}*",
        f"  {sig.signal_type}" + (f"  pattern=*{sig.pattern}*" if sig.pattern else ""),
        f"  {sig.symbol} {sig.timeframe}  direction: *{sig.direction}*",
        f"  bar: {sig.bar_timestamp}",
    ]
    if sig.entry_price is not None:
        lines.append(f"  entry: ${sig.entry_price:.2f}")
    if sig.target_price is not None:
        lines.append(f"  target: ${sig.target_price:.2f}")
    if sig.stop_price is not None:
        lines.append(f"  stop:   ${sig.stop_price:.2f}")
    if sig.ftfc_aligned is not None:
        lines.append(f"  FTFC: {'✓ aligned' if sig.ftfc_aligned else '✗ misaligned'}")
    if sig.recommended_size is not None:
        lines.append(f"  size: {sig.recommended_size:.1%} of capital")
    try:
        meta = json.loads(sig.metadata) if sig.metadata else {}
        if meta:
            extras = ", ".join(f"{k}={v}" for k, v in meta.items() if k != "trades")
            if extras:
                lines.append(f"  {extras}")
    except Exception:
        pass
    return "\n".join(lines)


# ── Telegram (with retry) ───────────────────────────────────────────────────

def _send_telegram_once(msg: str, timeout: float = OPENCLAW_TIMEOUT) -> None:
    """Single-attempt Telegram send. Raises on any failure."""
    payload = {"message": msg, "source": "strategy-engine"}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OPENCLAW_GATEWAY}/notify",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status >= 400:
            raise RuntimeError(f"gateway HTTP {resp.status}")


def send_telegram_with_retry(
    msg: str,
    *,
    max_retries: int = MAX_RETRIES,
    base_backoff_s: float = BASE_BACKOFF_S,
    timeout: float = OPENCLAW_TIMEOUT,
    sleep_fn=time.sleep,   # injectable for tests
) -> tuple[bool, int, str]:
    """
    Send with exponential backoff. Returns (success, attempts, last_error).
    """
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            _send_telegram_once(msg, timeout=timeout)
            return True, attempt, ""
        except Exception as e:  # noqa: BLE001 — we intentionally catch everything
            last_error = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                sleep_fn(base_backoff_s * (2 ** (attempt - 1)))
    return False, max_retries, last_error


# ── Notification log ────────────────────────────────────────────────────────

def _log_notification(
    signal_id: Optional[str],
    channel: str,
    status: str,
    attempts: int,
    elapsed_ms: int,
    error_message: str = "",
) -> None:
    """Persist a notification attempt. Best-effort — never raises."""
    try:
        con = duckdb.connect(str(LIVE_DB))
        try:
            con.execute(
                """
                INSERT INTO notification_log (
                    notif_id, signal_id, channel, attempted_at, attempts,
                    status, error_message, elapsed_ms
                ) VALUES (?, ?, ?, ?::TIMESTAMP, ?, ?, ?, ?)
                """,
                [
                    f"notif-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}",
                    signal_id,
                    channel,
                    datetime.now(timezone.utc).replace(tzinfo=None),
                    attempts,
                    status,
                    (error_message or "")[:500],
                    elapsed_ms,
                ],
            )
        finally:
            con.close()
    except Exception as log_err:  # don't let logging break the notification path
        print(f"  WARN: notification_log insert failed: {log_err}")


# ── Public API ──────────────────────────────────────────────────────────────

def send_telegram_signal(sig: SignalFired) -> bool:
    """Backwards-compatible: send a signal to Telegram, no retries exposed."""
    ok, _, _ = send_telegram_with_retry(format_signal(sig))
    return ok


def notify_signal(sig: SignalFired, channel: str = "auto") -> str:
    """
    Notify the user about a fired signal. Returns the channel used.

    channel:
      'auto'     — try telegram (with retries), fall back to stdout
      'telegram' — telegram only (raises if all retries fail)
      'stdout'   — print only
    """
    msg = format_signal(sig)

    if channel in ("auto", "telegram"):
        start = time.monotonic()
        ok, attempts, err = send_telegram_with_retry(msg)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        if ok:
            _log_notification(sig.signal_id, "telegram", "sent", attempts, elapsed_ms)
            return "telegram"
        if channel == "telegram":
            _log_notification(sig.signal_id, "telegram", "failed", attempts, elapsed_ms, err)
            raise RuntimeError(f"telegram notification failed after {attempts} attempts: {err}")
        # auto → fall back to stdout; log telegram failure first
        _log_notification(sig.signal_id, "telegram", "fell-back-stdout", attempts, elapsed_ms, err)

    # stdout fallback (or channel='stdout')
    start = time.monotonic()
    print(msg)
    elapsed_ms = int((time.monotonic() - start) * 1000)
    _log_notification(sig.signal_id, "stdout", "sent", 1, elapsed_ms)
    return "stdout"


def recent_notifications(hours: int = 24) -> list[dict]:
    """Return recent notification attempts for health-check / debugging."""
    con = duckdb.connect(str(LIVE_DB), read_only=True)
    try:
        cur = con.execute(
            """
            SELECT notif_id, signal_id, channel, attempted_at,
                   attempts, status, error_message, elapsed_ms
            FROM notification_log
            WHERE attempted_at >= CURRENT_TIMESTAMP - INTERVAL (?) HOUR
            ORDER BY attempted_at DESC
            """,
            [hours],
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()
