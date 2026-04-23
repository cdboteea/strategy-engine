"""
STRAT reversal pattern detection.

Given a DataFrame with classified bars (via `classification.classify_bars`),
scan for 6 patterns. Each pattern fires on the CURRENT bar — the third bar
in a 3-bar pattern (or second in a 2-bar quick reversal).

Patterns:
    2d-1-2u   down directional → inside → up directional       (bullish, medium)
    2u-1-2d   up directional   → inside → down directional     (bearish, medium)
    3-1-2u    outside         → inside → up directional       (bullish, high)
    3-1-2d    outside         → inside → down directional     (bearish, high)
    2d-2u     down directional → up directional               (bullish, medium, 2-bar)
    2u-2d     up directional   → down directional             (bearish, medium, 2-bar)

Output columns added to the classified DataFrame:
    strat_pattern   — pattern code if a pattern fires on this bar, else NaN
    strat_direction — 'bullish' or 'bearish' (None if no pattern)
"""
from __future__ import annotations
import pandas as pd

from .classification import TYPE_INSIDE, TYPE_DIRECTIONAL, TYPE_OUTSIDE


# Pattern specs: (pattern_code, direction, setup — list of (bar_type, direction) for bars [n-2, n-1, n])
# For 2-bar patterns, list is length 2 (bars [n-1, n])
_PATTERNS = [
    # 3-bar reversals (inside-bar pivot)
    ("2d-1-2u", "bullish",  [(TYPE_DIRECTIONAL, "down"), (TYPE_INSIDE, None), (TYPE_DIRECTIONAL, "up")]),
    ("2u-1-2d", "bearish",  [(TYPE_DIRECTIONAL, "up"),   (TYPE_INSIDE, None), (TYPE_DIRECTIONAL, "down")]),
    ("3-1-2u",  "bullish",  [(TYPE_OUTSIDE, None),       (TYPE_INSIDE, None), (TYPE_DIRECTIONAL, "up")]),
    ("3-1-2d",  "bearish",  [(TYPE_OUTSIDE, None),       (TYPE_INSIDE, None), (TYPE_DIRECTIONAL, "down")]),
    # 2-bar quick reversals (no inside bar)
    ("2d-2u",   "bullish",  [(TYPE_DIRECTIONAL, "down"), (TYPE_DIRECTIONAL, "up")]),
    ("2u-2d",   "bearish",  [(TYPE_DIRECTIONAL, "up"),   (TYPE_DIRECTIONAL, "down")]),
]


def _matches_setup(
    bar_types: list[int],
    directions: list[str],
    spec: list[tuple[int, str | None]],
) -> bool:
    """Check if the trailing bars match the pattern spec."""
    if len(bar_types) < len(spec) or len(directions) < len(spec):
        return False
    for i, (want_type, want_dir) in enumerate(spec):
        offset = len(spec) - 1 - i  # spec[0] is oldest
        bt = bar_types[-len(spec) + i]
        d = directions[-len(spec) + i]
        if bt is pd.NA or bt != want_type:
            return False
        if want_dir is not None and d != want_dir:
            return False
    return True


def detect_patterns(classified: pd.DataFrame) -> pd.DataFrame:
    """
    Scan classified bars for all 6 reversal patterns.

    Returns the input DataFrame with two new columns:
        strat_pattern    — pattern code ('2d-1-2u' etc.) or None
        strat_direction  — 'bullish' / 'bearish' / None
    """
    df = classified.copy()
    bar_types = df["bar_type"].tolist()
    directions = df["direction"].tolist()

    patterns: list[str | None] = [None] * len(df)
    pat_directions: list[str | None] = [None] * len(df)

    for i in range(len(df)):
        # Try each pattern, taking the first match (order matters — 3-bar before 2-bar
        # so a 2d-1-2u isn't also flagged as 2d-2u via the last 2 bars)
        for pat_code, pat_dir, spec in _PATTERNS:
            if _matches_setup(bar_types[: i + 1], directions[: i + 1], spec):
                patterns[i] = pat_code
                pat_directions[i] = pat_dir
                break

    df["strat_pattern"] = patterns
    df["strat_direction"] = pat_directions
    return df


def setup_bars_for_pattern(
    classified: pd.DataFrame,
    signal_idx: int,
    pattern_code: str,
) -> dict:
    """
    Return the setup bars (prior / inside / signal) for a pattern at a given index.

    For 3-bar patterns: {prior_bar, inside_bar, signal_bar}
    For 2-bar patterns: {prior_bar, signal_bar}

    Each value is the DataFrame row (as a dict with high/low/open/close).
    """
    is_3bar = pattern_code in ("2d-1-2u", "2u-1-2d", "3-1-2u", "3-1-2d")
    if is_3bar:
        if signal_idx < 2:
            return {}
        return {
            "prior_bar": classified.iloc[signal_idx - 2].to_dict(),
            "inside_bar": classified.iloc[signal_idx - 1].to_dict(),
            "signal_bar": classified.iloc[signal_idx].to_dict(),
        }
    else:
        if signal_idx < 1:
            return {}
        return {
            "prior_bar": classified.iloc[signal_idx - 1].to_dict(),
            "signal_bar": classified.iloc[signal_idx].to_dict(),
        }
