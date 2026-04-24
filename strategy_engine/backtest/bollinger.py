"""
Bollinger-mean-reversion strategy implementation.

Ported from ~/clawd/research/quant/scripts/spy-bollinger-{analysis,rebuild}.py
as a clean, parameterized function. The original SPY strategy is the canonical
test case (Sharpe 1.96, 86% win rate) — use this module to reproduce it and
then apply to other tickers.

Signal rule (matches spy-bollinger-hybrid-v1.yaml):
  - Compute 20-period SMA + 2σ Bollinger Bands on the close
  - Fire when close < lower band (optionally: close < prev lower band too)

Entry rule (hybrid-50-50):
  - First half of position entered at signal close
  - Second half at threshold (-5% for SPY, -7% for single stocks)
  - If threshold never hit, only the first half is held

Exit rule (profit-target):
  - First time the price hits signal_close * (1 + target) after entry → exit all
  - Otherwise exit at end of forward_window
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

import numpy as np
import pandas as pd

from ..config import BARS_PER_YEAR


@dataclass
class BollingerParams:
    lookback: int = 20
    std_dev: float = 2.0
    hybrid_second_half_depth: float = -0.05   # -5% for SPY, -7% for single names
    profit_target: float = 0.05               # +5%
    forward_window_bars: int = 13             # 13 weekly bars ≈ one quarter
    min_forward_bars: int = 11                # require at least 11 bars of forward data

    @classmethod
    def from_strategy(cls, strategy) -> "BollingerParams":
        """Extract params from a registry Strategy object."""
        sl = strategy.signal_logic
        entry = strategy.entry
        exit_ = strategy.exit

        # Access attrs from pydantic models (extra='allow' fields are attributes on v2)
        lookback = getattr(sl, "lookback", 20) or 20
        std_dev = getattr(sl, "std_dev", 2.0) or 2.0

        second_half = getattr(entry, "second_half", None) or {}
        if isinstance(second_half, dict):
            depth = second_half.get("depth", -0.05)
        else:
            depth = getattr(second_half, "depth", -0.05) or -0.05

        target = getattr(exit_, "target", 0.05) or 0.05
        forward_weeks = getattr(exit_, "forward_window_weeks", 13) or 13

        return cls(
            lookback=int(lookback),
            std_dev=float(std_dev),
            hybrid_second_half_depth=float(depth),
            profit_target=float(target),
            forward_window_bars=int(forward_weeks),
        )


@dataclass
class BollingerTrade:
    signal_date: pd.Timestamp
    signal_price: float
    first_half_entry_price: float
    second_half_entry_price: Optional[float]     # None if threshold not hit
    second_half_entry_date: Optional[pd.Timestamp]
    avg_entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_reason: str                              # "target" | "forward-window-end"
    holding_bars: int
    pct_return: float                             # relative to avg_entry_price
    target_hit: bool
    trough_pct: float                             # deepest drawdown within window


@dataclass
class BollingerResult:
    trades: list[BollingerTrade] = field(default_factory=list)
    # Per-trade summary
    n_signals: int = 0
    n_trades: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    median_return: float = 0.0
    profit_factor: float = 0.0
    total_pnl_pct: float = 0.0
    second_half_hit_rate: float = 0.0
    # Equity-curve metrics (the proper risk-adjusted ones)
    equity_sharpe: float = 0.0                     # annualized from full-bar equity returns
    equity_max_drawdown: float = 0.0               # deepest drawdown of equity curve (negative)
    equity_total_return: float = 0.0               # final equity / initial - 1
    equity_ann_return: float = 0.0                 # CAGR of equity curve
    # Active-bar Sharpe (excludes idle bars)
    active_bar_sharpe: float = 0.0
    active_bars: int = 0
    active_bar_fraction: float = 0.0
    # Worst per-trade trough (retained — different semantic from equity DD)
    worst_trade_trough: float = 0.0


def compute_bollinger(df: pd.DataFrame, lookback: int, std_dev: float) -> pd.DataFrame:
    """Add sma, upper, lower band columns to df. Requires 'close'."""
    df = df.copy()
    df["sma"] = df["close"].rolling(lookback).mean()
    df["std"] = df["close"].rolling(lookback).std(ddof=0)
    df["upper"] = df["sma"] + std_dev * df["std"]
    df["lower"] = df["sma"] - std_dev * df["std"]
    return df


def detect_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fire on EVERY bar where close < lower band. Each event is independent;
    back-to-back signals produce overlapping trades by design.

    Cluster analysis is downstream (post-hoc grouping by proximity), NOT a
    suppression rule on signal detection.
    """
    df = df.copy()
    df["below_lower"] = df["close"] < df["lower"]
    df["is_signal"] = df["below_lower"]
    return df


def simulate_trades(df: pd.DataFrame, params: BollingerParams) -> list[BollingerTrade]:
    """
    Given a df with signals (via detect_signals) + future bars, simulate
    hybrid-50-50 entry with profit-target exit.
    """
    df = df.sort_index()
    trades: list[BollingerTrade] = []
    signal_rows = df[df.get("is_signal", False)]

    for sig_ts, sig_row in signal_rows.iterrows():
        sig_price = float(sig_row["close"])
        target_price = sig_price * (1 + params.profit_target)
        threshold_price = sig_price * (1 + params.hybrid_second_half_depth)  # depth is negative

        # Forward window of subsequent bars
        future = df[df.index > sig_ts].head(params.forward_window_bars)
        if len(future) < params.min_forward_bars:
            continue  # not enough forward data

        first_half_entry = sig_price
        second_half_entry: Optional[float] = None
        second_half_date: Optional[pd.Timestamp] = None

        exit_date: Optional[pd.Timestamp] = None
        exit_price: Optional[float] = None
        exit_reason = ""
        trough = 0.0
        holding_bars = 0

        for fwd_ts, fwd_row in future.iterrows():
            holding_bars += 1
            fwd_high = float(fwd_row["high"])
            fwd_low = float(fwd_row["low"])
            fwd_close = float(fwd_row["close"])

            # Track deepest drawdown vs signal price (intra-trade)
            bar_trough_pct = (fwd_low - sig_price) / sig_price
            if bar_trough_pct < trough:
                trough = bar_trough_pct

            # Check second-half hybrid entry (if threshold hit and not yet in)
            if second_half_entry is None and fwd_low <= threshold_price:
                second_half_entry = threshold_price
                second_half_date = fwd_ts

            # Check profit-target exit
            if fwd_high >= target_price:
                exit_date = fwd_ts
                exit_price = target_price
                exit_reason = "target"
                break

        # If we didn't hit target, exit at end of forward window
        if exit_date is None:
            last = future.iloc[-1]
            exit_date = future.index[-1]
            exit_price = float(last["close"])
            exit_reason = "forward-window-end"

        # Compute avg_entry_price — 50/50 if second half hit, else 100% first half
        if second_half_entry is not None:
            avg_entry = (first_half_entry + second_half_entry) / 2
        else:
            avg_entry = first_half_entry

        pct_return = (exit_price - avg_entry) / avg_entry

        trades.append(
            BollingerTrade(
                signal_date=pd.Timestamp(sig_ts),
                signal_price=sig_price,
                first_half_entry_price=first_half_entry,
                second_half_entry_price=second_half_entry,
                second_half_entry_date=second_half_date,
                avg_entry_price=avg_entry,
                exit_date=pd.Timestamp(exit_date),
                exit_price=float(exit_price),
                exit_reason=exit_reason,
                holding_bars=holding_bars,
                pct_return=pct_return,
                target_hit=(exit_reason == "target"),
                trough_pct=trough,
            )
        )

    return trades


def build_equity_curve(
    trades: list[BollingerTrade],
    bars: pd.DataFrame,
    capital_allocation: float,
    round_trip_cost_pct: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Build a bar-level portfolio equity curve.

    Returns (equity, is_active) where:
      equity    — cumulative portfolio value (starts at 1.0)
      is_active — boolean series; True on bars with ≥1 open trade

    `round_trip_cost_pct` (e.g. 0.0008 = 8 bps) deducts the full round-trip
    cost from strategy_return at each trade's exit bar, sized by
    capital_allocation. Preserves intra-trade bar shape for drawdown
    metrics while correctly reducing cumulative equity.
    """
    is_active = pd.Series(False, index=bars.index)
    if not trades:
        return pd.Series(1.0, index=bars.index, name="equity"), is_active

    strategy_return = pd.Series(0.0, index=bars.index)
    for t in trades:
        try:
            slc = bars.loc[t.signal_date : t.exit_date]
        except KeyError:
            continue
        if len(slc) < 2:
            continue
        bar_returns = slc["close"].pct_change().dropna()
        strategy_return.loc[bar_returns.index] = (
            strategy_return.loc[bar_returns.index].values + bar_returns.values * capital_allocation
        )
        is_active.loc[slc.index] = True
        # Deduct round-trip cost at the exit bar (sized by allocation)
        if round_trip_cost_pct > 0 and t.exit_date in strategy_return.index:
            strategy_return.loc[t.exit_date] -= round_trip_cost_pct * capital_allocation

    equity = (1.0 + strategy_return).cumprod()
    equity.name = "equity"
    return equity, is_active


def _equity_metrics(
    equity: pd.Series,
    bars_per_year: int,
    is_active: Optional[pd.Series] = None,
) -> dict:
    """Sharpe + max DD + total return + CAGR + active-bar Sharpe."""
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
    if years > 0 and equity.iloc[0] > 0:
        ann_return = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)
    else:
        ann_return = 0.0

    # Active-bar Sharpe
    active_bar_sharpe = 0.0
    active_bars_count = 0
    active_frac = 0.0
    if is_active is not None and is_active.any():
        active_mask = is_active.reindex(returns.index).fillna(False)
        active_returns = returns[active_mask]
        active_bars_count = int(active_mask.sum())
        active_frac = float(active_mask.mean())
        if active_bars_count >= 2 and active_returns.std(ddof=0) > 0:
            active_bar_sharpe = float(
                active_returns.mean() / active_returns.std(ddof=0) * math.sqrt(bars_per_year)
            )

    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": total_return,
        "ann_return": ann_return,
        "active_bar_sharpe": active_bar_sharpe,
        "active_bars": active_bars_count,
        "active_bar_fraction": active_frac,
    }


def summarize(
    trades: list[BollingerTrade],
    bars: Optional[pd.DataFrame] = None,
    capital_allocation: float = 0.10,
    timeframe: str = "1w",
    cost_model=None,
) -> BollingerResult:
    """
    Compute per-trade + equity-curve summary metrics.

    `bars` + `capital_allocation` + `timeframe` must be passed to compute the
    equity-curve-based Sharpe and max drawdown. Without them (legacy call),
    equity metrics stay 0 and per-trade stats alone are reported.
    """
    if not trades:
        return BollingerResult()
    # Apply transaction cost model to each trade's pct_return BEFORE summary.
    # `trades` objects are mutated in place so the per-trade serialization
    # downstream (runner payload) reflects net returns.
    if cost_model is not None and cost_model.round_trip_pct > 0:
        for t in trades:
            t.pct_return = cost_model.apply_to_return(t.pct_return)
    returns = np.array([t.pct_return for t in trades])
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    worst_trough = min((t.trough_pct for t in trades), default=0.0)
    profit_factor = (
        float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    )
    second_half_hit = sum(1 for t in trades if t.second_half_entry_price is not None)

    result = BollingerResult(
        trades=trades,
        n_signals=len(trades),
        n_trades=len(trades),
        win_rate=float((returns > 0).mean()),
        avg_return=float(returns.mean()),
        median_return=float(np.median(returns)),
        profit_factor=profit_factor,
        total_pnl_pct=float(returns.sum()),
        second_half_hit_rate=second_half_hit / len(trades),
        worst_trade_trough=float(worst_trough),
    )

    if bars is not None:
        bars_per_year = BARS_PER_YEAR.get(timeframe, 252)
        round_trip_pct = cost_model.round_trip_pct if cost_model is not None else 0.0
        equity, is_active = build_equity_curve(
            trades, bars, capital_allocation, round_trip_cost_pct=round_trip_pct,
        )
        metrics = _equity_metrics(equity, bars_per_year, is_active=is_active)
        result.equity_sharpe = metrics["sharpe"]
        result.equity_max_drawdown = metrics["max_dd"]
        result.equity_total_return = metrics["total_return"]
        result.equity_ann_return = metrics["ann_return"]
        result.active_bar_sharpe = metrics["active_bar_sharpe"]
        result.active_bars = metrics["active_bars"]
        result.active_bar_fraction = metrics["active_bar_fraction"]

    return result


def run_bollinger(
    bars: pd.DataFrame,
    params: BollingerParams,
    *,
    capital_allocation: float = 0.10,
    timeframe: str = "1w",
    cost_model=None,
    regime_gate=None,
) -> BollingerResult:
    """
    End-to-end: compute bands → detect signals → (optional) regime-gate
    filter → simulate trades → summarize.

    `bars` must have a DatetimeIndex and columns [open, high, low, close, volume].
    For weekly-bar strategy, resample daily → weekly BEFORE calling this.

    `capital_allocation` and `timeframe` are used for the equity-curve Sharpe
    computation. `cost_model` (optional) nets transaction costs from each
    trade's pct_return. `regime_gate` (optional, e.g. VixGate) filters
    signals whose bar-date fails the gate condition — keeps only bars where
    the gate evaluates True.
    """
    if not {"open", "high", "low", "close"}.issubset(bars.columns):
        raise ValueError("bars must have open/high/low/close columns")
    df = compute_bollinger(bars, params.lookback, params.std_dev)
    df = detect_signals(df)

    # Regime-gate filter: drop `is_signal` at rows where the gate says no.
    # We apply BEFORE simulate_trades so filtered-out signals never enter the
    # trade list. This also makes the n_signals stat reflect post-gate count.
    if regime_gate is not None:
        from .regime import apply_vix_gate_to_signals
        sig_dates = df[df["is_signal"]].index.tolist()
        kept_dates, gate_stats = apply_vix_gate_to_signals(sig_dates, regime_gate)
        keep_set = set(kept_dates)
        df["is_signal"] = df.index.to_series().isin(keep_set) & df["is_signal"]

    trades = simulate_trades(df, params)
    result = summarize(
        trades, bars=bars, capital_allocation=capital_allocation,
        timeframe=timeframe, cost_model=cost_model,
    )
    return result
