"""
Momentum strategies — SMA crossover + MACD crossover.

SMA crossover:
  Fire BUY when fast SMA crosses above slow SMA (golden cross).
  Fire SELL when fast SMA crosses below slow SMA (death cross).
  Classic trend-following. Positive expectancy in trending regimes,
  negative in chop.

MACD crossover:
  Fire BUY when MACD line crosses above signal line AND histogram turns positive.
  Fire SELL on opposite cross.
  Variant of SMA with momentum acceleration filter via histogram.

Both strategies treat entries and exits symmetrically — entry on the
bullish cross, exit on the bearish cross (not profit target). This is
fundamentally different from the Bollinger/STRAT families which use
profit-target exits.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

import numpy as np
import pandas as pd

from ..config import BARS_PER_YEAR


# ═════════════════════════════════════════════════════════════════════
# SMA Crossover
# ═════════════════════════════════════════════════════════════════════

@dataclass
class SmaCrossoverParams:
    fast_window: int = 50
    slow_window: int = 200
    direction_bias: str = "long-only"   # "long-only" | "long-short" | "short-only"

    @classmethod
    def from_strategy(cls, strategy) -> "SmaCrossoverParams":
        sl = strategy.signal_logic
        return cls(
            fast_window=int(getattr(sl, "fast_window", 50) or 50),
            slow_window=int(getattr(sl, "slow_window", 200) or 200),
            direction_bias=str(getattr(sl, "direction_bias", "long-only") or "long-only"),
        )


@dataclass
class MomentumTrade:
    """Shared trade shape for SMA, MACD, Donchian, trend-pullback."""
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    direction: str           # "bullish" | "bearish"
    holding_bars: int
    pct_return: float
    exit_reason: str         # "reverse-cross" | "end-of-data" | "trailing-stop" | "pullback-resolved" | "trend-broken"
    symbol: Optional[str] = None   # populated for multi-ticker baskets


@dataclass
class MomentumResult:
    trades: list[MomentumTrade] = field(default_factory=list)
    n_signals: int = 0
    n_trades: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    median_return: float = 0.0
    profit_factor: float = 0.0
    total_pnl_pct: float = 0.0
    equity_sharpe: float = 0.0
    equity_max_drawdown: float = 0.0
    equity_total_return: float = 0.0
    equity_ann_return: float = 0.0
    active_bar_sharpe: float = 0.0
    active_bars: int = 0
    active_bar_fraction: float = 0.0


def compute_sma_crossover(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """Add sma_fast, sma_slow, and cross columns to df. Requires 'close'."""
    df = df.copy()
    df["sma_fast"] = df["close"].rolling(fast).mean()
    df["sma_slow"] = df["close"].rolling(slow).mean()
    df["spread"] = df["sma_fast"] - df["sma_slow"]
    prev_spread = df["spread"].shift(1)
    df["bullish_cross"] = (prev_spread <= 0) & (df["spread"] > 0)
    df["bearish_cross"] = (prev_spread >= 0) & (df["spread"] < 0)
    return df


def simulate_sma_crossover(df: pd.DataFrame, params: SmaCrossoverParams) -> list[MomentumTrade]:
    """
    Walk the bar series: each bullish_cross enters a long, each bearish_cross
    closes it (long-only mode). Exits use bar close.

    Works for any indicator that produces `bullish_cross` / `bearish_cross`
    boolean columns — SMA uses sma_slow, MACD uses signal_line. We check
    both and skip rows where the warm-up indicator is NaN.
    """
    trades: list[MomentumTrade] = []
    in_position = False
    entry_date = None
    entry_price = None
    entry_direction = None

    # Indicator warm-up check: skip any bar where the primary indicator is NaN.
    # SMA → sma_slow, MACD → signal_line. Use whichever is present.
    warmup_col = "sma_slow" if "sma_slow" in df.columns else (
        "signal_line" if "signal_line" in df.columns else None
    )

    for ts, row in df.iterrows():
        if warmup_col is not None and pd.isna(row.get(warmup_col)):
            continue
        bull = bool(row.get("bullish_cross", False))
        bear = bool(row.get("bearish_cross", False))

        if params.direction_bias == "long-only":
            if not in_position and bull:
                in_position = True
                entry_date = ts
                entry_price = float(row["close"])
                entry_direction = "bullish"
            elif in_position and bear:
                exit_price = float(row["close"])
                trades.append(MomentumTrade(
                    entry_date=entry_date, entry_price=entry_price,
                    exit_date=ts, exit_price=exit_price,
                    direction=entry_direction,
                    holding_bars=(df.index.get_loc(ts) - df.index.get_loc(entry_date)),
                    pct_return=(exit_price - entry_price) / entry_price,
                    exit_reason="reverse-cross",
                ))
                in_position = False
        elif params.direction_bias == "long-short":
            # Flip on each cross
            if bull:
                if in_position and entry_direction == "bearish":
                    exit_price = float(row["close"])
                    trades.append(MomentumTrade(
                        entry_date=entry_date, entry_price=entry_price,
                        exit_date=ts, exit_price=exit_price,
                        direction="bearish",
                        holding_bars=(df.index.get_loc(ts) - df.index.get_loc(entry_date)),
                        pct_return=(entry_price - exit_price) / entry_price,
                        exit_reason="reverse-cross",
                    ))
                in_position = True
                entry_date = ts
                entry_price = float(row["close"])
                entry_direction = "bullish"
            elif bear:
                if in_position and entry_direction == "bullish":
                    exit_price = float(row["close"])
                    trades.append(MomentumTrade(
                        entry_date=entry_date, entry_price=entry_price,
                        exit_date=ts, exit_price=exit_price,
                        direction="bullish",
                        holding_bars=(df.index.get_loc(ts) - df.index.get_loc(entry_date)),
                        pct_return=(exit_price - entry_price) / entry_price,
                        exit_reason="reverse-cross",
                    ))
                in_position = True
                entry_date = ts
                entry_price = float(row["close"])
                entry_direction = "bearish"
        # short-only mirrors long-only; omitted for v1

    # Close any open trade at the last bar
    if in_position and entry_date is not None:
        last = df.iloc[-1]
        exit_price = float(last["close"])
        pct = ((exit_price - entry_price) / entry_price) if entry_direction == "bullish" \
              else ((entry_price - exit_price) / entry_price)
        trades.append(MomentumTrade(
            entry_date=entry_date, entry_price=entry_price,
            exit_date=df.index[-1], exit_price=exit_price,
            direction=entry_direction,
            holding_bars=(len(df) - df.index.get_loc(entry_date) - 1),
            pct_return=pct,
            exit_reason="end-of-data",
        ))

    return trades


# ═════════════════════════════════════════════════════════════════════
# MACD Crossover
# ═════════════════════════════════════════════════════════════════════

@dataclass
class MacdCrossoverParams:
    fast_ema: int = 12
    slow_ema: int = 26
    signal_ema: int = 9
    require_histogram_positive: bool = True
    direction_bias: str = "long-only"

    @classmethod
    def from_strategy(cls, strategy) -> "MacdCrossoverParams":
        sl = strategy.signal_logic
        return cls(
            fast_ema=int(getattr(sl, "fast_ema", 12) or 12),
            slow_ema=int(getattr(sl, "slow_ema", 26) or 26),
            signal_ema=int(getattr(sl, "signal_ema", 9) or 9),
            require_histogram_positive=bool(
                getattr(sl, "require_histogram_positive", True)
            ),
            direction_bias=str(getattr(sl, "direction_bias", "long-only") or "long-only"),
        )


def compute_macd_crossover(df: pd.DataFrame, params: MacdCrossoverParams) -> pd.DataFrame:
    """Add macd_line, signal_line, histogram, and cross columns."""
    df = df.copy()
    ema_fast = df["close"].ewm(span=params.fast_ema, adjust=False).mean()
    ema_slow = df["close"].ewm(span=params.slow_ema, adjust=False).mean()
    df["macd_line"] = ema_fast - ema_slow
    df["signal_line"] = df["macd_line"].ewm(span=params.signal_ema, adjust=False).mean()
    df["histogram"] = df["macd_line"] - df["signal_line"]
    prev_hist = df["histogram"].shift(1)
    # Bullish MACD cross: histogram crosses from <=0 to >0
    df["bullish_cross"] = (prev_hist <= 0) & (df["histogram"] > 0)
    df["bearish_cross"] = (prev_hist >= 0) & (df["histogram"] < 0)
    return df


def simulate_macd_crossover(df: pd.DataFrame, params: MacdCrossoverParams) -> list[MomentumTrade]:
    """Same structure as SMA simulator (cross → entry, reverse cross → exit)."""
    # Reuse SMA simulator logic since the cross columns match
    sma_params = SmaCrossoverParams(direction_bias=params.direction_bias)
    return simulate_sma_crossover(df, sma_params)


# ═════════════════════════════════════════════════════════════════════
# Shared summarize (works for both SMA + MACD trades)
# ═════════════════════════════════════════════════════════════════════

def build_equity_curve(
    trades: list[MomentumTrade],
    bars: pd.DataFrame,
    capital_allocation: float,
    round_trip_cost_pct: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Bar-level equity curve with cost deducted at each trade's exit bar."""
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
        is_active.loc[slc.index] = True
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
    if len(equity) < 2:
        return {"sharpe": 0.0, "max_dd": 0.0, "total_return": 0.0, "ann_return": 0.0,
                "active_bar_sharpe": 0.0, "active_bars": 0, "active_bar_fraction": 0.0}

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
    ann_return = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0) \
                 if years > 0 and equity.iloc[0] > 0 else 0.0

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
        "sharpe": sharpe, "max_dd": max_dd, "total_return": total_return,
        "ann_return": ann_return,
        "active_bar_sharpe": active_bar_sharpe,
        "active_bars": active_bars_count,
        "active_bar_fraction": active_frac,
    }


def summarize(
    trades: list[MomentumTrade],
    bars: Optional[pd.DataFrame] = None,
    capital_allocation: float = 0.10,
    timeframe: str = "1d",
    cost_model=None,
) -> MomentumResult:
    if not trades:
        return MomentumResult()

    if cost_model is not None and cost_model.round_trip_pct > 0:
        for t in trades:
            t.pct_return = cost_model.apply_to_return(t.pct_return)

    returns = np.array([t.pct_return for t in trades])
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    profit_factor = (
        float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    )

    result = MomentumResult(
        trades=trades,
        n_signals=len(trades),
        n_trades=len(trades),
        win_rate=float((returns > 0).mean()),
        avg_return=float(returns.mean()),
        median_return=float(np.median(returns)),
        profit_factor=profit_factor,
        total_pnl_pct=float(returns.sum()),
    )

    if bars is not None:
        bars_per_year = BARS_PER_YEAR.get(timeframe, 252)
        round_trip_pct = cost_model.round_trip_pct if cost_model is not None else 0.0
        equity, is_active = build_equity_curve(
            trades, bars, capital_allocation, round_trip_cost_pct=round_trip_pct,
        )
        m = _equity_metrics(equity, bars_per_year, is_active=is_active)
        result.equity_sharpe = m["sharpe"]
        result.equity_max_drawdown = m["max_dd"]
        result.equity_total_return = m["total_return"]
        result.equity_ann_return = m["ann_return"]
        result.active_bar_sharpe = m["active_bar_sharpe"]
        result.active_bars = m["active_bars"]
        result.active_bar_fraction = m["active_bar_fraction"]

    return result


def _apply_regime_gate_to_crosses(df: pd.DataFrame, regime_gate) -> pd.DataFrame:
    """Zero out bullish_cross / bearish_cross on bars where the regime gate
    says NO. Preserves the `in_position exit` path — if we're already in a
    trade during a gated regime, the exit cross still fires.
    """
    if regime_gate is None:
        return df
    from .regime import apply_vix_gate_to_signals
    # Get bars where a bullish cross would fire
    bull_dates = df[df.get("bullish_cross", False)].index.tolist()
    kept, _ = apply_vix_gate_to_signals(bull_dates, regime_gate)
    keep_set = set(kept)
    df = df.copy()
    df["bullish_cross"] = df.index.to_series().isin(keep_set) & df["bullish_cross"]
    return df


def run_sma_crossover(
    bars: pd.DataFrame,
    params: SmaCrossoverParams,
    *,
    capital_allocation: float = 0.10,
    timeframe: str = "1d",
    cost_model=None,
    regime_gate=None,
) -> MomentumResult:
    df = compute_sma_crossover(bars, params.fast_window, params.slow_window)
    df = _apply_regime_gate_to_crosses(df, regime_gate)
    trades = simulate_sma_crossover(df, params)
    return summarize(trades, bars=bars, capital_allocation=capital_allocation,
                     timeframe=timeframe, cost_model=cost_model)


def run_macd_crossover(
    bars: pd.DataFrame,
    params: MacdCrossoverParams,
    *,
    capital_allocation: float = 0.10,
    timeframe: str = "1d",
    cost_model=None,
    regime_gate=None,
) -> MomentumResult:
    df = compute_macd_crossover(bars, params)
    df = _apply_regime_gate_to_crosses(df, regime_gate)
    trades = simulate_macd_crossover(df, params)
    return summarize(trades, bars=bars, capital_allocation=capital_allocation,
                     timeframe=timeframe, cost_model=cost_model)
