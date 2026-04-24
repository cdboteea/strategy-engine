"""
Breakout strategies — Donchian channel.

Fire BUY when close breaks above the rolling N-bar high.
Fire SELL when close breaks below the rolling N-bar low (long-only: exit).

Classic trend-following breakout (the Turtle system's core).

Exit rules:
  - Trailing stop at opposite-side channel low (for longs) / high (for shorts)
  - Or fixed max holding bars
  - Entry immediately on the breaking bar's close (signal = trigger)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..config import BARS_PER_YEAR
from .momentum import MomentumTrade, MomentumResult, build_equity_curve, _equity_metrics


@dataclass
class DonchianParams:
    entry_window: int = 20      # look-back for breakout trigger
    exit_window: int = 10       # look-back for trailing-stop exit
    direction_bias: str = "long-only"

    @classmethod
    def from_strategy(cls, strategy) -> "DonchianParams":
        sl = strategy.signal_logic
        return cls(
            entry_window=int(getattr(sl, "entry_window", 20) or 20),
            exit_window=int(getattr(sl, "exit_window", 10) or 10),
            direction_bias=str(getattr(sl, "direction_bias", "long-only") or "long-only"),
        )


def compute_donchian(df: pd.DataFrame, entry_window: int, exit_window: int) -> pd.DataFrame:
    """Add donchian_high, donchian_low (for entry), and exit_low (for trailing stop)."""
    df = df.copy()
    # Use the PREVIOUS N bars (shift by 1) so the current bar's close can break it
    df["donchian_high"] = df["high"].rolling(entry_window).max().shift(1)
    df["donchian_low"] = df["low"].rolling(entry_window).min().shift(1)
    df["exit_low"] = df["low"].rolling(exit_window).min().shift(1)
    df["exit_high"] = df["high"].rolling(exit_window).max().shift(1)
    df["bullish_breakout"] = df["close"] > df["donchian_high"]
    df["bearish_breakout"] = df["close"] < df["donchian_low"]
    return df


def simulate_donchian(df: pd.DataFrame, params: DonchianParams) -> list[MomentumTrade]:
    """Long-only Donchian: enter on bullish_breakout, exit on close < exit_low."""
    trades: list[MomentumTrade] = []
    in_position = False
    entry_date = None
    entry_price = None
    entry_direction = None

    for ts, row in df.iterrows():
        if pd.isna(row.get("donchian_high")) or pd.isna(row.get("exit_low")):
            continue

        if params.direction_bias == "long-only":
            if not in_position and bool(row.get("bullish_breakout", False)):
                in_position = True
                entry_date = ts
                entry_price = float(row["close"])
                entry_direction = "bullish"
            elif in_position:
                # Exit when close < trailing exit_low
                if float(row["close"]) < float(row["exit_low"]):
                    exit_price = float(row["close"])
                    trades.append(MomentumTrade(
                        entry_date=entry_date, entry_price=entry_price,
                        exit_date=ts, exit_price=exit_price,
                        direction=entry_direction,
                        holding_bars=(df.index.get_loc(ts) - df.index.get_loc(entry_date)),
                        pct_return=(exit_price - entry_price) / entry_price,
                        exit_reason="trailing-stop",
                    ))
                    in_position = False

    if in_position and entry_date is not None:
        last = df.iloc[-1]
        exit_price = float(last["close"])
        trades.append(MomentumTrade(
            entry_date=entry_date, entry_price=entry_price,
            exit_date=df.index[-1], exit_price=exit_price,
            direction=entry_direction,
            holding_bars=(len(df) - df.index.get_loc(entry_date) - 1),
            pct_return=(exit_price - entry_price) / entry_price,
            exit_reason="end-of-data",
        ))

    return trades


def summarize(
    trades: list[MomentumTrade],
    bars: Optional[pd.DataFrame] = None,
    capital_allocation: float = 0.10,
    timeframe: str = "1d",
    cost_model=None,
) -> MomentumResult:
    """Reuse momentum.summarize — same result shape."""
    from .momentum import summarize as m_summarize
    return m_summarize(trades, bars=bars, capital_allocation=capital_allocation,
                       timeframe=timeframe, cost_model=cost_model)


def run_donchian(
    bars: pd.DataFrame,
    params: DonchianParams,
    *,
    capital_allocation: float = 0.10,
    timeframe: str = "1d",
    cost_model=None,
) -> MomentumResult:
    if not {"open", "high", "low", "close"}.issubset(bars.columns):
        raise ValueError("bars must have OHLC columns")
    df = compute_donchian(bars, params.entry_window, params.exit_window)
    trades = simulate_donchian(df, params)
    return summarize(trades, bars=bars, capital_allocation=capital_allocation,
                     timeframe=timeframe, cost_model=cost_model)
