"""
Full Timeframe Continuity (FTFC).

For each bar on the trade timeframe, look up the current bar on higher
timeframes (1mo, 1w, 1d, 1h by default) and compute per-timeframe directional
bias (bullish if close > open, bearish if close < open, neutral if equal).

A trade-timeframe bar has "bullish FTFC" if the fraction of aligned bullish
higher-tf bars meets the threshold (default 0.75 = 3 of 4).

Implementation:
- Each higher-timeframe DataFrame is reindexed onto the trade-timeframe's
  DatetimeIndex using ffill (each trade bar picks up the most-recent-completed
  higher-tf bar).
- Directional bias per bar: sign(close - open)
- FTFC score = fraction of higher-tfs aligned bullish/bearish
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd


def _bar_direction(df: pd.DataFrame) -> pd.Series:
    """+1 bullish, -1 bearish, 0 neutral."""
    diff = df["close"] - df["open"]
    direction = pd.Series(0, index=df.index, dtype="int8")
    direction[diff > 0] = 1
    direction[diff < 0] = -1
    return direction


def align_higher_tf_to_trade_tf(
    trade_bars: pd.DataFrame,
    higher_tf_bars: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reindex `higher_tf_bars` onto `trade_bars.index`, forward-filling values.

    A trade bar at timestamp T will carry the higher-tf bar with label <= T
    (i.e. the most recently completed higher-tf bar as of T).
    """
    if higher_tf_bars.empty:
        return pd.DataFrame(index=trade_bars.index, columns=higher_tf_bars.columns)
    return higher_tf_bars.reindex(trade_bars.index, method="ffill")


def compute_ftfc(
    trade_bars: pd.DataFrame,
    higher_timeframes: dict[str, pd.DataFrame],
    *,
    threshold: float = 0.75,
) -> pd.DataFrame:
    """
    Compute per-bar FTFC score and flags.

    Args:
        trade_bars: OHLCV at the strategy's trade timeframe (DatetimeIndex)
        higher_timeframes: {tf_name: OHLCV DataFrame} — e.g. {'1mo': monthly_df, '1w': weekly_df, '1d': daily_df}
                          Include the trade timeframe itself here (e.g. '1h' if strategy is 1h).
        threshold: fraction of aligned bars needed (default 0.75)

    Returns:
        DataFrame on trade_bars.index with columns:
            ftfc_bullish_score  — fraction of higher-tfs bullish
            ftfc_bearish_score  — fraction of higher-tfs bearish
            is_bullish_ftfc     — bool, bullish_score >= threshold
            is_bearish_ftfc     — bool, bearish_score >= threshold
            ftfc_<tf>           — directional int per timeframe
    """
    if not higher_timeframes:
        raise ValueError("Need at least one higher timeframe for FTFC")

    out = pd.DataFrame(index=trade_bars.index)
    per_tf_directions: list[pd.Series] = []

    for tf_name, tf_bars in higher_timeframes.items():
        if tf_bars.empty:
            # No data for this timeframe — mark all neutral
            aligned_dir = pd.Series(0, index=trade_bars.index, dtype="int8", name=tf_name)
        else:
            # Compute direction on the HIGHER tf first, then align
            htf_direction = _bar_direction(tf_bars)
            aligned_dir = htf_direction.reindex(trade_bars.index, method="ffill")
            aligned_dir = aligned_dir.fillna(0).astype("int8")
            aligned_dir.name = tf_name
        out[f"ftfc_{tf_name}"] = aligned_dir
        per_tf_directions.append(aligned_dir)

    stacked = pd.concat(per_tf_directions, axis=1)
    n_tf = stacked.shape[1]

    bullish_count = (stacked > 0).sum(axis=1)
    bearish_count = (stacked < 0).sum(axis=1)

    out["ftfc_bullish_score"] = bullish_count / n_tf
    out["ftfc_bearish_score"] = bearish_count / n_tf
    out["is_bullish_ftfc"] = out["ftfc_bullish_score"] >= threshold
    out["is_bearish_ftfc"] = out["ftfc_bearish_score"] >= threshold

    return out
