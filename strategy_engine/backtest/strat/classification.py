"""
Bar classification per Rob Smith's "The STRAT".

Scenario 1 (Inside):  current high <= prev high AND current low >= prev low
Scenario 2 (Directional):
  - 2u: current high >  prev high AND current low >= prev low  (breaks up only)
  - 2d: current low  <  prev low  AND current high <= prev high  (breaks down only)
Scenario 3 (Outside): current high >  prev high AND current low <  prev low

Vectorized over a DataFrame with columns [high, low] and a DatetimeIndex.
"""
from __future__ import annotations
import pandas as pd


# Bar type codes (match the-strat-implementation-plan.md)
TYPE_INSIDE = 1
TYPE_DIRECTIONAL = 2
TYPE_OUTSIDE = 3


def classify_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Add bar-classification columns to `bars`.

    Output columns added:
        prev_high, prev_low      — previous bar's high/low (NaN on first row)
        bar_type                 — 1 (inside), 2 (directional), 3 (outside), NaN for first row
        direction                — 'up', 'down', or 'neutral' (inside/outside)
        breaks_high, breaks_low  — bool flags

    First row has NaN classification (no previous bar to compare).
    """
    if not {"high", "low"}.issubset(bars.columns):
        raise ValueError("bars must have high and low columns")

    df = bars.copy()
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)

    breaks_high = df["high"] > df["prev_high"]
    breaks_low = df["low"] < df["prev_low"]

    # Scenario 3 (outside): both breaks
    is_outside = breaks_high & breaks_low
    # Scenario 2 up: breaks high only
    is_2u = breaks_high & ~breaks_low
    # Scenario 2 down: breaks low only
    is_2d = breaks_low & ~breaks_high
    # Scenario 1 (inside): neither
    is_inside = ~breaks_high & ~breaks_low

    bar_type = pd.Series(pd.NA, index=df.index, dtype="Int64")
    bar_type[is_outside] = TYPE_OUTSIDE
    bar_type[is_2u | is_2d] = TYPE_DIRECTIONAL
    bar_type[is_inside] = TYPE_INSIDE
    df["bar_type"] = bar_type

    direction = pd.Series("neutral", index=df.index, dtype="object")
    direction[is_2u] = "up"
    direction[is_2d] = "down"
    direction[is_outside] = "outside"
    # First row has no prev → mark neutral; downstream code should check bar_type for NA
    direction.iloc[0] = "neutral"
    df["direction"] = direction

    df["breaks_high"] = breaks_high.fillna(False)
    df["breaks_low"] = breaks_low.fillna(False)

    return df
