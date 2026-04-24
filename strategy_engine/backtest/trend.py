"""
Trend-following — 200-SMA pullback.

Thesis: when the market is in a clear uptrend (close > 200-SMA), pullbacks
to short-term oversold (close < 20-SMA) are buying opportunities. Exit
when the pullback is resolved (close > 20-SMA again) OR the trend breaks
(close < 200-SMA).

Philosophy is opposite to Bollinger mean-reversion: Bollinger assumes
prices return to mean; this assumes prices continue the trend after
temporary deviation.

Signal logic:
  - In uptrend:   close > 200-SMA
  - Entry:        close < 20-SMA AND in uptrend
  - Exit A:       close > 20-SMA   (pullback resolved)
  - Exit B:       close < 200-SMA  (trend broken — hard stop)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .momentum import MomentumTrade, MomentumResult


@dataclass
class TrendPullbackParams:
    long_sma: int = 200       # trend filter
    short_sma: int = 20       # pullback filter
    direction_bias: str = "long-only"

    @classmethod
    def from_strategy(cls, strategy) -> "TrendPullbackParams":
        sl = strategy.signal_logic
        return cls(
            long_sma=int(getattr(sl, "long_sma", 200) or 200),
            short_sma=int(getattr(sl, "short_sma", 20) or 20),
            direction_bias=str(getattr(sl, "direction_bias", "long-only") or "long-only"),
        )


def compute_trend_pullback(df: pd.DataFrame, long_sma: int, short_sma: int) -> pd.DataFrame:
    df = df.copy()
    df["sma_long"] = df["close"].rolling(long_sma).mean()
    df["sma_short"] = df["close"].rolling(short_sma).mean()
    df["in_uptrend"] = df["close"] > df["sma_long"]
    df["in_pullback"] = df["close"] < df["sma_short"]
    df["entry_condition"] = df["in_uptrend"] & df["in_pullback"]
    return df


def simulate_trend_pullback(df: pd.DataFrame, params: TrendPullbackParams) -> list[MomentumTrade]:
    trades: list[MomentumTrade] = []
    in_position = False
    entry_date = None
    entry_price = None

    for ts, row in df.iterrows():
        if pd.isna(row.get("sma_long")):
            continue

        if not in_position and bool(row.get("entry_condition", False)):
            in_position = True
            entry_date = ts
            entry_price = float(row["close"])
        elif in_position:
            close = float(row["close"])
            # Exit A: pullback resolved (close back above short SMA)
            if close > float(row["sma_short"]):
                trades.append(MomentumTrade(
                    entry_date=entry_date, entry_price=entry_price,
                    exit_date=ts, exit_price=close,
                    direction="bullish",
                    holding_bars=(df.index.get_loc(ts) - df.index.get_loc(entry_date)),
                    pct_return=(close - entry_price) / entry_price,
                    exit_reason="pullback-resolved",
                ))
                in_position = False
            # Exit B: trend broken (close below long SMA)
            elif not bool(row.get("in_uptrend", False)):
                trades.append(MomentumTrade(
                    entry_date=entry_date, entry_price=entry_price,
                    exit_date=ts, exit_price=close,
                    direction="bullish",
                    holding_bars=(df.index.get_loc(ts) - df.index.get_loc(entry_date)),
                    pct_return=(close - entry_price) / entry_price,
                    exit_reason="trend-broken",
                ))
                in_position = False

    if in_position and entry_date is not None:
        last = df.iloc[-1]
        close = float(last["close"])
        trades.append(MomentumTrade(
            entry_date=entry_date, entry_price=entry_price,
            exit_date=df.index[-1], exit_price=close,
            direction="bullish",
            holding_bars=(len(df) - df.index.get_loc(entry_date) - 1),
            pct_return=(close - entry_price) / entry_price,
            exit_reason="end-of-data",
        ))

    return trades


def run_trend_pullback(
    bars: pd.DataFrame,
    params: TrendPullbackParams,
    *,
    capital_allocation: float = 0.10,
    timeframe: str = "1d",
    cost_model=None,
) -> MomentumResult:
    from .momentum import summarize as m_summarize
    if not {"close"}.issubset(bars.columns):
        raise ValueError("bars must have close column")
    df = compute_trend_pullback(bars, params.long_sma, params.short_sma)
    trades = simulate_trend_pullback(df, params)
    return m_summarize(trades, bars=bars, capital_allocation=capital_allocation,
                       timeframe=timeframe, cost_model=cost_model)
