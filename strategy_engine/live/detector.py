"""
Live signal detector.

For each promoted strategy, fetches the latest bars, runs the backtest logic,
and checks if a signal fires on the most recent completed bar. If yes:
  - Writes to live_signals DuckDB
  - Sends notification (optional)
  - Returns a SignalFired record

Reuses 100% of the backtest logic — no duplicated signal-detection code.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import json
import uuid

import duckdb
import pandas as pd

from .. import __version__ as ENGINE_VERSION
from ..providers.duckdb_provider import load_ohlcv, load_multi_timeframe
from ..registry.loader import load_one, validate_all
from ..registry.schema import Strategy
from ..backtest import bollinger as bol
from ..backtest.strat import (
    classify_bars,
    detect_patterns,
    compute_ftfc,
    StratParams,
    simulate_trades as strat_simulate,
)


LIVE_DB = Path.home() / "clawd" / "data" / "live-signals.duckdb"


@dataclass
class SignalFired:
    signal_id: str
    strategy_id: str
    fired_at: str
    bar_timestamp: str
    symbol: str
    timeframe: str
    signal_type: str
    pattern: Optional[str]
    direction: str
    ftfc_aligned: Optional[bool]
    entry_price: Optional[float]
    stop_price: Optional[float]
    target_price: Optional[float]
    recommended_size: Optional[float]
    metadata: str   # JSON


def _new_signal_id(strategy_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"sig-{strategy_id}-{ts}-{uuid.uuid4().hex[:6]}"


def _detect_bollinger(strategy: Strategy) -> Optional[SignalFired]:
    """Run Bollinger logic on the latest bars. Return SignalFired if the
    latest bar's close is below the lower band."""
    symbol = strategy.instruments[0]
    tf = strategy.timeframe
    bars = load_ohlcv(symbol, tf)
    if bars.empty or len(bars) < 25:
        return None

    params = bol.BollingerParams.from_strategy(strategy)
    classified = bol.compute_bollinger(bars, params.lookback, params.std_dev)
    classified = bol.detect_signals(classified)

    latest = classified.iloc[-1]
    if not bool(latest.get("is_signal", False)):
        return None

    close = float(latest["close"])
    lower = float(latest["lower"])
    # Entry is the signal close (first half of hybrid); second-half threshold is -5% or -7%
    second_half = getattr(strategy.entry, "second_half", None) or {}
    if isinstance(second_half, dict):
        depth = float(second_half.get("depth", -0.05))
    else:
        depth = float(getattr(second_half, "depth", -0.05))
    threshold_price = close * (1 + depth)
    target = close * (1 + float(getattr(strategy.exit, "target", 0.05)))

    return SignalFired(
        signal_id=_new_signal_id(strategy.id),
        strategy_id=strategy.id,
        fired_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        bar_timestamp=str(classified.index[-1]),
        symbol=symbol,
        timeframe=tf,
        signal_type="bollinger-lower-band",
        pattern=None,
        direction="bullish",
        ftfc_aligned=None,
        entry_price=close,
        stop_price=None,      # Bollinger hybrid doesn't use a fixed stop; forward-window exit
        target_price=target,
        recommended_size=strategy.capital_allocation,
        metadata=json.dumps({
            "lower_band": lower,
            "sma": float(latest["sma"]),
            "second_half_trigger_price": threshold_price,
            "forward_window_weeks": int(getattr(strategy.exit, "forward_window_weeks", 13)),
        }),
    )


def _detect_strat(strategy: Strategy) -> Optional[SignalFired]:
    """Run STRAT logic on the latest bars. Return SignalFired if the wanted
    pattern fires on the latest completed bar AND FTFC is aligned (if required)."""
    symbol = strategy.instruments[0]
    tf = strategy.timeframe
    sl = strategy.signal_logic

    bars = load_ohlcv(symbol, tf)
    if bars.empty or len(bars) < 3:
        return None

    classified = detect_patterns(classify_bars(bars))
    wanted = getattr(sl, "pattern", None)
    if wanted:
        classified["strat_pattern"] = classified["strat_pattern"].where(
            classified["strat_pattern"] == wanted, None
        )

    latest = classified.iloc[-1]
    pattern = latest.get("strat_pattern")
    if not pattern:
        return None
    direction = latest.get("strat_direction", "bullish")

    # FTFC check
    ftfc_aligned = True
    ftfc_score = 1.0
    if bool(getattr(sl, "require_ftfc", True)):
        ftfc_tfs = list(getattr(sl, "ftfc_timeframes", ["1mo", "1w", "1d", "1h"]))
        all_tfs = list(dict.fromkeys([*ftfc_tfs, tf]))
        htfs = load_multi_timeframe(symbol, all_tfs)
        ftfc = compute_ftfc(bars, htfs, threshold=float(getattr(sl, "ftfc_threshold", 0.75)))
        if bars.index[-1] in ftfc.index:
            r = ftfc.loc[bars.index[-1]]
            if direction == "bullish":
                ftfc_aligned = bool(r.get("is_bullish_ftfc", False))
                ftfc_score = float(r.get("ftfc_bullish_score", 0.0))
            else:
                ftfc_aligned = bool(r.get("is_bearish_ftfc", False))
                ftfc_score = float(r.get("ftfc_bearish_score", 0.0))
        if not ftfc_aligned:
            return None

    # Compute entry/target/stop — reuse the simulator helper for consistency
    from ..backtest.strat.simulator import _compute_trade_levels
    sig_idx = classified.index.get_loc(bars.index[-1])
    levels = _compute_trade_levels(classified, sig_idx, pattern, direction)
    if levels is None:
        return None
    entry_trigger, target_price, stop_price = levels

    return SignalFired(
        signal_id=_new_signal_id(strategy.id),
        strategy_id=strategy.id,
        fired_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        bar_timestamp=str(classified.index[-1]),
        symbol=symbol,
        timeframe=tf,
        signal_type="strat-pattern",
        pattern=pattern,
        direction=direction,
        ftfc_aligned=ftfc_aligned,
        entry_price=entry_trigger,
        stop_price=stop_price,
        target_price=target_price,
        recommended_size=strategy.capital_allocation,
        metadata=json.dumps({
            "ftfc_score": ftfc_score,
            "ftfc_threshold": float(getattr(sl, "ftfc_threshold", 0.75)),
            "confidence": getattr(sl, "confidence", "medium"),
        }),
    )


def detect_signals_for_strategy(strategy_id: str) -> Optional[SignalFired]:
    """Check one strategy. Returns a SignalFired if the latest bar triggers."""
    from ..backtest.runner import _find_yaml_path
    yaml_path = _find_yaml_path(strategy_id)
    if not yaml_path:
        raise ValueError(f"No YAML for {strategy_id!r}")
    strategy = load_one(yaml_path)

    signal_type = strategy.signal_logic.type
    if signal_type == "bollinger-mean-reversion":
        return _detect_bollinger(strategy)
    elif signal_type == "strat-pattern":
        return _detect_strat(strategy)
    else:
        raise ValueError(f"Detector doesn't support {signal_type!r}")


def persist_signal(sig: SignalFired) -> None:
    """Insert SignalFired into live-signals.duckdb (idempotent on signal_id)."""
    LIVE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(LIVE_DB))
    try:
        existing = con.execute(
            "SELECT 1 FROM live_signals WHERE signal_id = ?", [sig.signal_id]
        ).fetchone()
        if existing:
            return
        con.execute(
            """
            INSERT INTO live_signals (
                signal_id, strategy_id, fired_at, bar_timestamp, symbol, timeframe,
                signal_type, pattern, direction, ftfc_aligned,
                entry_price, stop_price, target_price, recommended_size,
                notification_sent, status, engine_version, metadata
            ) VALUES (?, ?, ?::TIMESTAMP, ?::TIMESTAMP, ?, ?,
                     ?, ?, ?, ?,
                     ?, ?, ?, ?,
                     FALSE, 'new', ?, ?)
            """,
            [
                sig.signal_id, sig.strategy_id, sig.fired_at, sig.bar_timestamp,
                sig.symbol, sig.timeframe, sig.signal_type, sig.pattern,
                sig.direction, sig.ftfc_aligned,
                sig.entry_price, sig.stop_price, sig.target_price, sig.recommended_size,
                ENGINE_VERSION, sig.metadata,
            ],
        )
    finally:
        con.close()


def _is_due_now(timeframe: str, now: Optional[datetime] = None) -> bool:
    """
    Return True if we should check this timeframe at `now` (defaults to now).

    Rough schedule (US equities):
      1h  — top of every hour during market hours
      4h  — 13:30 ET and 16:01 ET (and optional 09:30 open check)
      1d  — 16:01 ET (after close)
      1w  — Friday 16:01 ET
      1mo — last trading day of month 16:01 ET (approximated: 1d + month-end check)

    Callers can override by passing --timeframe; this function is used when
    running `detect --all-promoted` with auto-scheduling.
    """
    now = now or datetime.now().astimezone()
    hour = now.hour
    minute = now.minute
    wday = now.weekday()  # Mon=0 Sun=6

    # Only consider US equity trading hours Mon-Fri
    if wday >= 5:
        return False

    if timeframe == "1h":
        return minute < 5  # Top of hour, 5-min window
    if timeframe == "4h":
        return (hour == 13 and minute < 45) or (hour == 16 and minute < 5)
    if timeframe == "1d":
        return hour == 16 and minute < 15
    if timeframe == "1w":
        return wday == 4 and hour == 16 and minute < 15  # Friday 16:00+
    if timeframe == "1mo":
        # Approximate: last trading day of month, 16:00
        next_day = (now + pd.Timedelta(days=1)).day
        is_last = next_day == 1 or (wday == 4 and (now + pd.Timedelta(days=3)).month != now.month)
        return is_last and hour == 16 and minute < 15
    return False


def detect_all_promoted(
    *,
    persist: bool = True,
    respect_schedule: bool = True,
) -> list[SignalFired]:
    """
    Iterate all strategies with status='promoted' and check each.
    When respect_schedule=True, skip strategies whose timeframe isn't due now.

    Returns a list of fired signals.
    """
    strategies, _errors = validate_all()
    promoted = [s for s in strategies if s.status == "promoted"]
    fired: list[SignalFired] = []

    for strat in promoted:
        if respect_schedule and not _is_due_now(strat.timeframe):
            continue
        try:
            sig = detect_signals_for_strategy(strat.id)
        except Exception as e:
            print(f"  ERROR {strat.id}: {e}", flush=True)
            continue
        if sig is None:
            continue
        fired.append(sig)
        if persist:
            persist_signal(sig)
    return fired
