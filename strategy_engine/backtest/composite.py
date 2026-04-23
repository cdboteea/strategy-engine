"""
Composite strategy backtest.

A composite takes one `primary` strategy plus one or more `confirmations`.
The engine:
  1. Runs the primary end-to-end (full trade simulation)
  2. Runs each confirmation and collects their signal events
  3. Filters primary trades: keep those whose signal_date has a matching
     confirmation signal within ±window_days (mode='any' or 'all',
     with optional direction-match requirement)
  4. Re-summarizes the filtered trade list using primary's equity-curve logic

v1 constraints:
  - Primary must be `bollinger-mean-reversion`
  - Confirmations can be `strat-pattern` or `bollinger-mean-reversion`
  - Result shape matches BollingerResult (plus composite filter stats)

This keeps the engine small: the composite is a *filter* on top of existing
strategies, not a new signal generator.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ..config import BARS_PER_YEAR
from ..providers.duckdb_provider import load_ohlcv, load_multi_timeframe, DataNotAvailable
from ..registry.loader import load_one
from ..registry.schema import Strategy
from . import bollinger as bol
from .strat import (
    classify_bars,
    detect_patterns,
    compute_ftfc,
    StratParams,
    simulate_trades as strat_simulate,
)


class CompositeError(Exception):
    pass


@dataclass
class SignalEvent:
    """Normalized signal record, agnostic to source strategy type."""
    date: pd.Timestamp
    direction: str  # 'bullish' | 'bearish'
    source_id: str
    source_type: str  # 'bollinger-mean-reversion' | 'strat-pattern'


@dataclass
class CompositeFilterStats:
    """How many primary trades survived confirmation filtering."""
    primary_trades_raw: int = 0
    primary_trades_after_confirmation: int = 0
    confirmations_per_source: dict[str, int] = field(default_factory=dict)
    drop_reason_counts: dict[str, int] = field(default_factory=dict)


def _find_yaml_path(strategy_id: str):
    from ..config import REGISTRY_DIR
    matches = list(REGISTRY_DIR.rglob(f"{strategy_id}.yaml"))
    return matches[0] if matches else None


def _load_strategy_by_id(strategy_id: str) -> Strategy:
    path = _find_yaml_path(strategy_id)
    if not path:
        raise CompositeError(f"Composite references unknown strategy id {strategy_id!r}")
    return load_one(path)


# -- Event extractors -------------------------------------------------------

def _run_primary_bollinger(
    strategy: Strategy,
    start: Optional[str],
    end: Optional[str],
    *,
    cost_model=None,
) -> tuple[bol.BollingerResult, pd.DataFrame]:
    """Run a Bollinger primary end-to-end. Returns the full result + bars."""
    if strategy.signal_logic.type != "bollinger-mean-reversion":
        raise CompositeError(
            f"composite v1 requires primary=bollinger-mean-reversion, "
            f"got {strategy.signal_logic.type!r} for {strategy.id}"
        )
    symbol = strategy.instruments[0]
    tf = strategy.timeframe
    bars = load_ohlcv(symbol=symbol, timeframe=tf, start=start, end=end)
    if bars.empty:
        raise DataNotAvailable(f"No bars for primary {symbol} {tf}")
    params = bol.BollingerParams.from_strategy(strategy)
    result = bol.run_bollinger(
        bars, params,
        capital_allocation=strategy.capital_allocation,
        timeframe=tf,
        cost_model=cost_model,
    )
    return result, bars


def _events_from_bollinger(
    strategy: Strategy,
    start: Optional[str],
    end: Optional[str],
) -> list[SignalEvent]:
    result, _ = _run_primary_bollinger(strategy, start, end)
    # Bollinger signals are always bullish mean-reversion entries.
    return [
        SignalEvent(
            date=pd.Timestamp(t.signal_date),
            direction="bullish",
            source_id=strategy.id,
            source_type="bollinger-mean-reversion",
        )
        for t in result.trades
    ]


def _events_from_strat(
    strategy: Strategy,
    start: Optional[str],
    end: Optional[str],
) -> list[SignalEvent]:
    """Run a STRAT strategy and return one SignalEvent per (FTFC-passing) trade."""
    symbol = strategy.instruments[0]
    tf = strategy.timeframe
    sl = strategy.signal_logic

    trade_bars = load_ohlcv(symbol=symbol, timeframe=tf, start=start, end=end)
    if trade_bars.empty:
        return []

    classified = classify_bars(trade_bars)
    classified = detect_patterns(classified)

    wanted_pattern = getattr(sl, "pattern", None)
    if wanted_pattern:
        classified["strat_pattern"] = classified["strat_pattern"].where(
            classified["strat_pattern"] == wanted_pattern, None
        )

    require_ftfc = bool(getattr(sl, "require_ftfc", True))
    ftfc_timeframes = list(getattr(sl, "ftfc_timeframes", ["1mo", "1w", "1d", "1h"]))
    ftfc_threshold = float(getattr(sl, "ftfc_threshold", 0.75))

    ftfc_df = None
    if require_ftfc:
        all_tfs = list(dict.fromkeys([*ftfc_timeframes, tf]))
        try:
            htfs = load_multi_timeframe(symbol, all_tfs, start=start, end=end)
            ftfc_df = compute_ftfc(trade_bars, htfs, threshold=ftfc_threshold)
        except DataNotAvailable:
            return []

    params = StratParams(
        require_ftfc=require_ftfc,
        ftfc_threshold=ftfc_threshold,
        min_risk_reward=float(getattr(strategy.exit, "min_risk_reward", 1.5)),
        max_holding_bars=int(getattr(sl, "max_holding_bars", 20)),
    )

    trades, _n_raw, _n_ftfc = strat_simulate(classified, ftfc_df, params)
    return [
        SignalEvent(
            date=pd.Timestamp(t.signal_date),
            direction=t.direction,
            source_id=strategy.id,
            source_type="strat-pattern",
        )
        for t in trades
    ]


def _events_from_strategy(
    strategy: Strategy,
    start: Optional[str],
    end: Optional[str],
) -> list[SignalEvent]:
    stype = strategy.signal_logic.type
    if stype == "bollinger-mean-reversion":
        return _events_from_bollinger(strategy, start, end)
    if stype == "strat-pattern":
        return _events_from_strat(strategy, start, end)
    raise CompositeError(
        f"Confirmation strategy {strategy.id} has unsupported type {stype!r} "
        f"(composite v1 supports: bollinger-mean-reversion, strat-pattern)"
    )


# -- Filtering --------------------------------------------------------------

def _matches_within_window(
    primary_event: SignalEvent,
    confirm_events: list[SignalEvent],
    window_days: int,
    require_direction_match: bool,
) -> bool:
    """Return True if any confirm event falls within ±window_days of primary."""
    if not confirm_events:
        return False
    delta = pd.Timedelta(days=window_days)
    lo = primary_event.date - delta
    hi = primary_event.date + delta
    for ce in confirm_events:
        if ce.date < lo or ce.date > hi:
            continue
        if require_direction_match and ce.direction != primary_event.direction:
            continue
        return True
    return False


def filter_primary_trades(
    primary_trades: list,   # list[BollingerTrade]
    confirmations: dict[str, list[SignalEvent]],  # source_id -> events
    mode: str,
    window_days: int,
    require_direction_match: bool,
) -> tuple[list, CompositeFilterStats]:
    """
    Keep only primary trades confirmed by `mode` rule across the given
    confirmation event streams.

    mode='any'  — at least one confirmation hits within window
    mode='all'  — every confirmation source must hit within window
    """
    stats = CompositeFilterStats(
        primary_trades_raw=len(primary_trades),
        confirmations_per_source={k: len(v) for k, v in confirmations.items()},
    )
    kept: list = []
    for t in primary_trades:
        # Primary trades are always treated as bullish for Bollinger mean-reversion
        pev = SignalEvent(
            date=pd.Timestamp(t.signal_date),
            direction="bullish",
            source_id="__primary__",
            source_type="bollinger-mean-reversion",
        )
        hits_per_source = {
            src: _matches_within_window(pev, evs, window_days, require_direction_match)
            for src, evs in confirmations.items()
        }
        if mode == "all":
            passed = all(hits_per_source.values())
            reason = None if passed else "missing-one-or-more-confirmations"
        elif mode == "any":
            passed = any(hits_per_source.values())
            reason = None if passed else "no-confirmation-in-window"
        else:
            raise CompositeError(f"Unknown composite mode {mode!r}")
        if passed:
            kept.append(t)
        else:
            stats.drop_reason_counts[reason] = stats.drop_reason_counts.get(reason, 0) + 1

    stats.primary_trades_after_confirmation = len(kept)
    return kept, stats


# -- End-to-end runner ------------------------------------------------------

@dataclass
class CompositeRunResult:
    """BollingerResult with composite filter metadata attached."""
    result: bol.BollingerResult
    primary_bars: pd.DataFrame
    primary_strategy: Strategy
    filter_stats: CompositeFilterStats


def run_composite(
    strategy: Strategy,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    cost_model=None,
) -> CompositeRunResult:
    """
    Run a composite strategy end-to-end and return summary metrics +
    filter-stage diagnostics. Cost model applies to the final filtered
    trade list (not to the discarded raw-primary trades).
    """
    if strategy.composite is None or strategy.signal_logic.type != "composite":
        raise CompositeError(f"{strategy.id}: not a composite strategy")

    cm = strategy.composite

    # 1. Run primary end-to-end WITHOUT costs — we'll apply costs at the
    #    re-summarize step on the filtered trade list only.
    primary = _load_strategy_by_id(cm.primary)
    primary_result, primary_bars = _run_primary_bollinger(primary, start, end, cost_model=None)

    # 2. Collect confirmation events
    confirmations: dict[str, list[SignalEvent]] = {}
    for conf_id in cm.confirmations:
        conf_strat = _load_strategy_by_id(conf_id)
        confirmations[conf_id] = _events_from_strategy(conf_strat, start, end)

    # 3. Filter primary trades
    filtered_trades, stats = filter_primary_trades(
        primary_result.trades,
        confirmations,
        mode=cm.mode,
        window_days=cm.window_days,
        require_direction_match=cm.require_direction_match,
    )

    # 4. Re-summarize filtered trades with the composite's own cost model
    final = bol.summarize(
        filtered_trades,
        bars=primary_bars,
        capital_allocation=strategy.capital_allocation,
        timeframe=primary.timeframe,
        cost_model=cost_model,
    )

    return CompositeRunResult(
        result=final,
        primary_bars=primary_bars,
        primary_strategy=primary,
        filter_stats=stats,
    )
