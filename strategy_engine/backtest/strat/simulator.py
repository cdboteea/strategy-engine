"""
STRAT trade simulator.

For each bar where a pattern fires:
  - Determine entry trigger (break of inside-bar high for bullish, low for bearish)
  - Determine target (prior-bar high/low)
  - Determine stop (inside-bar low/high, or for 2-bar patterns the signal bar's opposite)
  - Optionally gate on FTFC (require_ftfc)
  - Simulate forward: if stop hit first → loss; if target hit → win; else exit at max_holding_bars

Entry simulation is simplified:
  - We enter on the NEXT bar if the pattern fires AND (optionally) FTFC aligns
  - Entry price = trigger level (we assume a stop-limit order to capture the breakout)
  - Stop price = inside-bar opposite extreme (for 3-bar) or signal-bar opposite extreme (for 2-bar)
  - Target price = prior-bar extreme

For 2-bar patterns (2d-2u, 2u-2d):
  - prior_bar == bar before signal (typically a directional bar)
  - Target = prior bar's high (bullish) or low (bearish)
  - Stop = signal bar's opposite extreme
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

import numpy as np
import pandas as pd

from ...config import BARS_PER_YEAR
from .patterns import setup_bars_for_pattern


@dataclass
class StratParams:
    require_ftfc: bool = True
    ftfc_threshold: float = 0.75
    min_risk_reward: float = 1.5
    max_holding_bars: int = 20


@dataclass
class StratTrade:
    pattern: str
    direction: str                   # 'bullish' or 'bearish'
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    target_price: float
    stop_price: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_reason: str                 # 'target' | 'stop' | 'window-end'
    holding_bars: int
    pct_return: float
    risk_reward_planned: float       # ratio at entry
    ftfc_aligned: bool
    ftfc_score: float


@dataclass
class StratResult:
    trades: list[StratTrade] = field(default_factory=list)
    # Per-trade summary
    n_signals_raw: int = 0            # before FTFC filter
    n_signals_ftfc: int = 0           # after FTFC filter
    n_trades: int = 0                 # executed (may drop those failing min_risk_reward)
    win_rate: float = 0.0
    avg_return: float = 0.0
    profit_factor: float = 0.0
    total_pnl_pct: float = 0.0
    # Equity curve (full-bar — includes idle bars)
    equity_sharpe: float = 0.0
    equity_max_drawdown: float = 0.0
    equity_total_return: float = 0.0
    equity_ann_return: float = 0.0
    # Active-bar Sharpe — computed only on bars where ≥1 trade is open.
    # Higher than full-bar Sharpe because idle-bar noise is excluded. Useful
    # for comparing "quality of trades when taken" across strategies with
    # very different activation rates.
    active_bar_sharpe: float = 0.0
    active_bars: int = 0              # count of bars with ≥1 open trade
    active_bar_fraction: float = 0.0  # active_bars / total_bars in backtest window
    # Exits breakdown
    target_hit_rate: float = 0.0
    stop_hit_rate: float = 0.0


def _compute_trade_levels(
    classified: pd.DataFrame,
    signal_idx: int,
    pattern: str,
    direction: str,
) -> Optional[tuple[float, float, float]]:
    """
    Compute (entry_trigger, target, stop) from the pattern's setup bars.

    Per the-strat-implementation-plan.md §2.2:
      3-bar (2d-1-2u, 2u-1-2d, 3-1-2u, 3-1-2d):
        entry  = inside-bar high (bullish) / low (bearish)     — break of inside
        stop   = inside-bar low  (bullish) / high (bearish)
        target = prior-bar high  (bullish) / low (bearish)
      2-bar (2d-2u, 2u-2d):
        entry  = prior-bar high  (bullish) / low (bearish)     — break of the prior directional bar
        stop   = prior-bar low   (bullish) / high (bearish)
        target = bar-before-prior high (bullish) / low (bearish), fallback prior * 1.02 / 0.98

    Returns None if setup bars can't be resolved.
    """
    bars = setup_bars_for_pattern(classified, signal_idx, pattern)
    if not bars:
        return None

    is_3bar = "inside_bar" in bars
    prior_bar = bars["prior_bar"]

    if is_3bar:
        inside_bar = bars["inside_bar"]
        if direction == "bullish":
            entry = inside_bar["high"]
            stop = inside_bar["low"]
            target = prior_bar["high"]
        else:
            entry = inside_bar["low"]
            stop = inside_bar["high"]
            target = prior_bar["low"]
    else:
        # 2-bar quick reversal. Entry/stop are on the PRIOR bar (the directional
        # bar being reversed). Target is bar-before-prior (one further back).
        if signal_idx >= 2:
            pre_prior = classified.iloc[signal_idx - 2].to_dict()
        else:
            pre_prior = None
        if direction == "bullish":
            entry = prior_bar["high"]
            stop = prior_bar["low"]
            target = float(pre_prior["high"]) if pre_prior is not None else float(prior_bar["high"]) * 1.02
        else:
            entry = prior_bar["low"]
            stop = prior_bar["high"]
            target = float(pre_prior["low"]) if pre_prior is not None else float(prior_bar["low"]) * 0.98

    return (float(entry), float(target), float(stop))


def simulate_trades(
    classified: pd.DataFrame,
    ftfc: Optional[pd.DataFrame],
    params: StratParams,
) -> tuple[list[StratTrade], int, int]:
    """
    Run forward simulation. Returns (trades, n_signals_raw, n_signals_ftfc).
    """
    trades: list[StratTrade] = []
    n_raw = 0
    n_ftfc = 0

    signal_rows = classified[classified["strat_pattern"].notna()]

    for sig_ts, sig_row in signal_rows.iterrows():
        n_raw += 1
        pattern = sig_row["strat_pattern"]
        direction = sig_row["strat_direction"]
        sig_idx = classified.index.get_loc(sig_ts)

        # FTFC gate
        ftfc_aligned = True
        ftfc_score = 1.0
        if params.require_ftfc and ftfc is not None:
            if sig_ts not in ftfc.index:
                continue
            ftfc_row = ftfc.loc[sig_ts]
            if direction == "bullish":
                ftfc_aligned = bool(ftfc_row.get("is_bullish_ftfc", False))
                ftfc_score = float(ftfc_row.get("ftfc_bullish_score", 0.0))
            else:
                ftfc_aligned = bool(ftfc_row.get("is_bearish_ftfc", False))
                ftfc_score = float(ftfc_row.get("ftfc_bearish_score", 0.0))
            if not ftfc_aligned:
                continue
        n_ftfc += 1

        # Trade levels
        levels = _compute_trade_levels(classified, sig_idx, pattern, direction)
        if levels is None:
            continue
        entry_trigger, target_price, stop_price = levels

        # Compute risk/reward from entry trigger
        if direction == "bullish":
            risk = entry_trigger - stop_price
            reward = target_price - entry_trigger
        else:
            risk = stop_price - entry_trigger
            reward = entry_trigger - target_price

        if risk <= 0:
            continue
        rr = reward / risk
        if rr < params.min_risk_reward:
            continue

        # Forward-sim from the bar AFTER the signal
        forward_bars = classified.iloc[sig_idx + 1 : sig_idx + 1 + params.max_holding_bars]
        if forward_bars.empty:
            continue

        entry_date: Optional[pd.Timestamp] = None
        entry_price: Optional[float] = None
        exit_date: Optional[pd.Timestamp] = None
        exit_price: Optional[float] = None
        exit_reason = ""

        for fwd_ts, fwd_row in forward_bars.iterrows():
            fwd_high = float(fwd_row["high"])
            fwd_low = float(fwd_row["low"])
            fwd_close = float(fwd_row["close"])

            # Stage 1: wait for entry trigger
            if entry_date is None:
                if direction == "bullish" and fwd_high >= entry_trigger:
                    entry_date = fwd_ts
                    entry_price = entry_trigger  # assume stop-limit fills at trigger
                elif direction == "bearish" and fwd_low <= entry_trigger:
                    entry_date = fwd_ts
                    entry_price = entry_trigger
                # If both trigger and stop hit in same bar, assume entry then stop
                if entry_date == fwd_ts and entry_price is not None:
                    # Check stop/target intra-bar
                    if direction == "bullish":
                        if fwd_low <= stop_price:
                            exit_date = fwd_ts
                            exit_price = stop_price
                            exit_reason = "stop"
                            break
                        if fwd_high >= target_price:
                            exit_date = fwd_ts
                            exit_price = target_price
                            exit_reason = "target"
                            break
                    else:
                        if fwd_high >= stop_price:
                            exit_date = fwd_ts
                            exit_price = stop_price
                            exit_reason = "stop"
                            break
                        if fwd_low <= target_price:
                            exit_date = fwd_ts
                            exit_price = target_price
                            exit_reason = "target"
                            break
                continue

            # Stage 2: in trade — check stop/target
            if direction == "bullish":
                if fwd_low <= stop_price:
                    exit_date = fwd_ts
                    exit_price = stop_price
                    exit_reason = "stop"
                    break
                if fwd_high >= target_price:
                    exit_date = fwd_ts
                    exit_price = target_price
                    exit_reason = "target"
                    break
            else:
                if fwd_high >= stop_price:
                    exit_date = fwd_ts
                    exit_price = stop_price
                    exit_reason = "stop"
                    break
                if fwd_low <= target_price:
                    exit_date = fwd_ts
                    exit_price = target_price
                    exit_reason = "target"
                    break

        # Never entered?
        if entry_date is None:
            continue

        # Never exited? Close at last forward bar
        if exit_date is None:
            last = forward_bars.iloc[-1]
            exit_date = forward_bars.index[-1]
            exit_price = float(last["close"])
            exit_reason = "window-end"

        # P&L
        if direction == "bullish":
            pct_return = (exit_price - entry_price) / entry_price
        else:
            pct_return = (entry_price - exit_price) / entry_price

        trades.append(StratTrade(
            pattern=pattern,
            direction=direction,
            signal_date=pd.Timestamp(sig_ts),
            entry_date=pd.Timestamp(entry_date),
            entry_price=float(entry_price),
            target_price=float(target_price),
            stop_price=float(stop_price),
            exit_date=pd.Timestamp(exit_date),
            exit_price=float(exit_price),
            exit_reason=exit_reason,
            holding_bars=int((classified.index.get_loc(exit_date) - classified.index.get_loc(entry_date))),
            pct_return=float(pct_return),
            risk_reward_planned=float(rr),
            ftfc_aligned=ftfc_aligned,
            ftfc_score=ftfc_score,
        ))

    return trades, n_raw, n_ftfc


def build_equity_curve(
    trades: list[StratTrade],
    bars: pd.DataFrame,
    capital_allocation: float,
) -> tuple[pd.Series, pd.Series]:
    """
    Bar-level equity curve.

    Returns (equity, is_active) where:
      equity    — cumulative portfolio value (starts at 1.0)
      is_active — boolean series, True on bars with ≥1 open trade
                  (used for active-bar Sharpe computation)
    """
    is_active = pd.Series(False, index=bars.index)
    if not trades:
        return pd.Series(1.0, index=bars.index, name="equity"), is_active

    strategy_return = pd.Series(0.0, index=bars.index)
    for t in trades:
        try:
            slc = bars.loc[t.entry_date : t.exit_date]
        except KeyError:
            continue
        if len(slc) < 2:
            continue
        bar_returns = slc["close"].pct_change().dropna()
        if t.direction == "bearish":
            bar_returns = -bar_returns
        strategy_return.loc[bar_returns.index] = (
            strategy_return.loc[bar_returns.index].values + bar_returns.values * capital_allocation
        )
        # Mark bars during this trade as active (including entry bar)
        is_active.loc[slc.index] = True

    equity = (1.0 + strategy_return).cumprod()
    equity.name = "equity"
    return equity, is_active


def _equity_metrics(
    equity: pd.Series,
    bars_per_year: int,
    is_active: Optional[pd.Series] = None,
) -> dict:
    if len(equity) < 2:
        return {
            "sharpe": 0.0, "max_dd": 0.0, "total_return": 0.0, "ann_return": 0.0,
            "active_bar_sharpe": 0.0, "active_bars": 0, "active_bar_fraction": 0.0,
        }

    returns = equity.pct_change().dropna()
    if returns.std(ddof=0) == 0 or len(returns) == 0:
        sharpe = 0.0
    else:
        sharpe = float(returns.mean() / returns.std(ddof=0) * math.sqrt(bars_per_year))

    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    max_dd = float(dd.min()) if len(dd) else 0.0

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    n_bars = len(equity)
    years = n_bars / bars_per_year if bars_per_year > 0 else 1.0
    ann_return = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0) if years > 0 else 0.0

    # Active-bar Sharpe: Sharpe computed only over bars where ≥1 trade is open.
    # Same annualization factor (bars_per_year) — we're abstracting from idle time.
    active_bar_sharpe = 0.0
    active_bars = 0
    active_frac = 0.0
    if is_active is not None and is_active.any():
        # Returns aligned to the is_active mask (returns has one fewer row than equity)
        active_mask = is_active.reindex(returns.index).fillna(False)
        active_returns = returns[active_mask]
        active_bars = int(active_mask.sum())
        active_frac = float(active_mask.mean())
        if active_bars >= 2 and active_returns.std(ddof=0) > 0:
            active_bar_sharpe = float(
                active_returns.mean() / active_returns.std(ddof=0) * math.sqrt(bars_per_year)
            )

    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": total_return,
        "ann_return": ann_return,
        "active_bar_sharpe": active_bar_sharpe,
        "active_bars": active_bars,
        "active_bar_fraction": active_frac,
    }


def summarize(
    trades: list[StratTrade],
    n_signals_raw: int,
    n_signals_ftfc: int,
    bars: pd.DataFrame,
    capital_allocation: float,
    timeframe: str,
    cost_model=None,
) -> StratResult:
    if not trades:
        return StratResult(
            n_signals_raw=n_signals_raw,
            n_signals_ftfc=n_signals_ftfc,
        )

    # Apply transaction cost to each trade's pct_return in place
    if cost_model is not None and cost_model.round_trip_pct > 0:
        for t in trades:
            t.pct_return = cost_model.apply_to_return(t.pct_return)

    returns = np.array([t.pct_return for t in trades])
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    profit_factor = (
        float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    )
    target_hits = sum(1 for t in trades if t.exit_reason == "target")
    stop_hits = sum(1 for t in trades if t.exit_reason == "stop")

    bars_per_year = BARS_PER_YEAR.get(timeframe, 252)
    equity, is_active = build_equity_curve(trades, bars, capital_allocation)
    metrics = _equity_metrics(equity, bars_per_year, is_active=is_active)

    return StratResult(
        trades=trades,
        n_signals_raw=n_signals_raw,
        n_signals_ftfc=n_signals_ftfc,
        n_trades=len(trades),
        win_rate=float((returns > 0).mean()),
        avg_return=float(returns.mean()),
        profit_factor=profit_factor,
        total_pnl_pct=float(returns.sum()),
        equity_sharpe=metrics["sharpe"],
        equity_max_drawdown=metrics["max_dd"],
        equity_total_return=metrics["total_return"],
        equity_ann_return=metrics["ann_return"],
        active_bar_sharpe=metrics["active_bar_sharpe"],
        active_bars=metrics["active_bars"],
        active_bar_fraction=metrics["active_bar_fraction"],
        target_hit_rate=target_hits / len(trades),
        stop_hit_rate=stop_hits / len(trades),
    )
