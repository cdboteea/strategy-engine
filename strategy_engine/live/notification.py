"""
Signal notification channels.

Primary: Telegram via OpenClaw gateway (port 18789).
Fallback: stdout.
"""
from __future__ import annotations
from typing import Optional
import json
import subprocess
import urllib.request
import urllib.error

from .detector import SignalFired


OPENCLAW_GATEWAY = "http://127.0.0.1:18789"
OPENCLAW_TIMEOUT = 3.0


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


def send_telegram_signal(sig: SignalFired) -> bool:
    """
    Send signal to Telegram via OpenClaw gateway.
    Returns True on success, False otherwise (caller can fall back to stdout).
    """
    msg = format_signal(sig)
    payload = {"message": msg, "source": "strategy-engine"}

    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{OPENCLAW_GATEWAY}/notify",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=OPENCLAW_TIMEOUT) as resp:
            return resp.status < 400
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def notify_signal(sig: SignalFired, channel: str = "auto") -> str:
    """
    Notify the user about a fired signal. Returns the channel used.

    channel:
      'auto'     — try telegram, fall back to stdout
      'telegram' — telegram only (raises if unavailable)
      'stdout'   — print only
    """
    if channel in ("auto", "telegram"):
        if send_telegram_signal(sig):
            return "telegram"
        if channel == "telegram":
            raise RuntimeError("telegram notification failed")
    # stdout fallback
    print(format_signal(sig))
    return "stdout"
