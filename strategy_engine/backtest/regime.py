"""
Regime-gate primitive.

A regime gate is a condition on a broad-market indicator (VIX, breadth, rates)
that must be satisfied for a strategy to fire a signal. Used to suspend
mean-reversion strategies during panic-volatility windows where the
"bounce" takes longer than the strategy's forward window.

v1 supports:
  - VIX level gate — skip trades when VIX is above (or below) a threshold
  - Regime-state gate — skip trades when VIX state matches a labeled regime

Future extensions (not v1):
  - Breadth gate (% stocks above 200-SMA, McClellan oscillator)
  - Rates gate (10Y yield level or curve shape)
  - Custom gate plugins

Design: gates evaluate at bar-level. Given a date, return True (fire OK) or
False (suppress). Applied in the signal-detection path AFTER the primary
signal condition fires — a signal that would have fired is filtered out
if the regime gate is False.

The gate has a historical data requirement: to backtest with the gate,
we need VIX (or other index) bars aligned to the strategy's timeframe.
The provider handles this via firstrate (historical) + EODHD (fresh).
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Literal

import pandas as pd
import duckdb


FIRSTRATE_DB = Path.home() / "clawd" / "data" / "firstrate.duckdb"


# ── VIX loader ─────────────────────────────────────────────────────────────

def load_vix_daily(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Load VIX daily bars from firstrate.duckdb.

    For dates past firstrate's ceiling, caller should splice EODHD separately.
    Returns a DataFrame with DatetimeIndex and 'close' column (at minimum).
    """
    con = duckdb.connect(str(FIRSTRATE_DB), read_only=True)
    try:
        sql = """
            SELECT datetime, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = 'VIX' AND timeframe = 'day'
        """
        params: list = []
        if start is not None:
            sql += " AND datetime >= ?"
            params.append(str(start))
        if end is not None:
            sql += " AND datetime <= ?"
            params.append(str(end))
        sql += " ORDER BY datetime"
        df = con.execute(sql, params).df()
    finally:
        con.close()
    if df.empty:
        return df
    df = df.set_index("datetime")
    return df


# ── Gate definitions ──────────────────────────────────────────────────────

@dataclass
class VixGate:
    """Suppress trades based on VIX level.

    mode = 'below'  — fire only when VIX_close < threshold  (skip panic vol)
    mode = 'above'  — fire only when VIX_close > threshold  (contrarian)
    mode = 'between'— fire only when lower <= VIX_close <= upper

    Typical mean-reversion use: VixGate(threshold=35, mode='below') — skip
    any bollinger signal fired when VIX > 35, because the 13-week forward
    window is unlikely to mean-revert during an ongoing panic.
    """
    mode: Literal["below", "above", "between"] = "below"
    threshold: Optional[float] = None          # for 'below' / 'above'
    lower: Optional[float] = None              # for 'between'
    upper: Optional[float] = None              # for 'between'

    def __post_init__(self):
        if self.mode == "between":
            if self.lower is None or self.upper is None:
                raise ValueError("VixGate(mode='between') requires lower AND upper")
        else:
            if self.threshold is None:
                raise ValueError(f"VixGate(mode='{self.mode}') requires threshold")

    def evaluate(self, vix_close: float) -> bool:
        """Return True if trade should fire, False to suppress."""
        if self.mode == "below":
            return vix_close < self.threshold
        if self.mode == "above":
            return vix_close > self.threshold
        if self.mode == "between":
            return self.lower <= vix_close <= self.upper
        raise ValueError(f"unknown VixGate mode {self.mode!r}")

    def describe(self) -> str:
        if self.mode == "below":
            return f"VIX < {self.threshold}"
        if self.mode == "above":
            return f"VIX > {self.threshold}"
        if self.mode == "between":
            return f"{self.lower} <= VIX <= {self.upper}"
        return f"VIX({self.mode})"


# ── Apply gate to signal dates ──────────────────────────────────────────

@dataclass
class GateApplicationStats:
    """Diagnostics from applying a gate to a signal list."""
    n_signals_in: int
    n_signals_out: int
    n_dropped: int
    n_no_vix_data: int = 0              # signals on dates with no VIX bar


def apply_vix_gate_to_signals(
    signal_dates: list[pd.Timestamp],
    gate: VixGate,
    vix_bars: Optional[pd.DataFrame] = None,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> tuple[list[pd.Timestamp], GateApplicationStats]:
    """Filter a list of signal dates through a VixGate.

    Returns (kept_signals, stats). A signal is kept if, on the trading day
    at or before the signal_date, VIX satisfies the gate condition.
    (If the signal bar has no matching VIX bar — rare — we KEEP the signal
    conservatively and report n_no_vix_data.)

    `vix_bars` may be pre-loaded; if None, we load from firstrate.
    """
    if vix_bars is None:
        vix_bars = load_vix_daily(start=start, end=end)

    stats = GateApplicationStats(
        n_signals_in=len(signal_dates),
        n_signals_out=0,
        n_dropped=0,
    )

    if vix_bars is None or vix_bars.empty:
        stats.n_no_vix_data = len(signal_dates)
        stats.n_signals_out = len(signal_dates)
        return list(signal_dates), stats

    kept: list[pd.Timestamp] = []
    vix_close = vix_bars["close"].sort_index()

    for sd in signal_dates:
        sd_ts = pd.Timestamp(sd)
        # Find the most recent VIX bar on-or-before sd
        idx = vix_close.index.searchsorted(sd_ts, side="right") - 1
        if idx < 0:
            stats.n_no_vix_data += 1
            kept.append(sd)  # conservative: keep if no VIX data
            continue
        vix_val = float(vix_close.iloc[idx])
        if gate.evaluate(vix_val):
            kept.append(sd)

    stats.n_signals_out = len(kept)
    stats.n_dropped = stats.n_signals_in - stats.n_signals_out
    return kept, stats


# ── Registry schema integration helper ─────────────────────────────────

def gate_from_config(config: dict | None) -> Optional[VixGate]:
    """Build a VixGate from a YAML config dict (or None).

    Config shape (optional `regime_gate` block on a strategy YAML):

        regime_gate:
          type: vix
          mode: below             # or above, between
          threshold: 35
          # OR for between:
          # lower: 15
          # upper: 30

    Returns None if config is None/empty.
    """
    if not config:
        return None
    if config.get("type", "vix") != "vix":
        raise ValueError(
            f"regime_gate.type = {config.get('type')!r} not supported (v1: only 'vix')"
        )
    mode = config.get("mode", "below")
    if mode in ("below", "above"):
        return VixGate(mode=mode, threshold=float(config["threshold"]))
    if mode == "between":
        return VixGate(mode=mode, lower=float(config["lower"]), upper=float(config["upper"]))
    raise ValueError(f"regime_gate.mode = {mode!r} not supported (v1: below/above/between)")
