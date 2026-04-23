"""Behavioral tests for STRAT modules — classification, FTFC, patterns, simulator."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from strategy_engine.backtest.strat import (
    classify_bars, TYPE_INSIDE, TYPE_DIRECTIONAL, TYPE_OUTSIDE,
    compute_ftfc,
    detect_patterns,
    StratParams, simulate_trades, summarize,
)


# ─── classification ─────────────────────────────────────────────────────────


def test_classify_inside_bar():
    df = pd.DataFrame({
        "open":  [100, 102],
        "high":  [105, 104],   # 104 < 105 → inside
        "low":   [95,  98],    # 98 > 95  → inside
        "close": [102, 103],
        "volume": [1e6, 1e6],
    }, index=pd.date_range("2020-01-01", periods=2, freq="D"))
    out = classify_bars(df)
    assert out.iloc[1]["bar_type"] == TYPE_INSIDE
    assert out.iloc[1]["direction"] == "neutral"


def test_classify_2u_directional():
    df = pd.DataFrame({
        "open":  [100, 103],
        "high":  [105, 108],   # breaks up
        "low":   [95,  100],   # doesn't break down (100 > 95)
        "close": [102, 107],
        "volume": [1e6, 1e6],
    }, index=pd.date_range("2020-01-01", periods=2, freq="D"))
    out = classify_bars(df)
    assert out.iloc[1]["bar_type"] == TYPE_DIRECTIONAL
    assert out.iloc[1]["direction"] == "up"


def test_classify_2d_directional():
    df = pd.DataFrame({
        "open":  [100, 98],
        "high":  [105, 102],   # doesn't break up
        "low":   [95,  90],    # breaks down
        "close": [102, 93],
        "volume": [1e6, 1e6],
    }, index=pd.date_range("2020-01-01", periods=2, freq="D"))
    out = classify_bars(df)
    assert out.iloc[1]["bar_type"] == TYPE_DIRECTIONAL
    assert out.iloc[1]["direction"] == "down"


def test_classify_outside_bar():
    df = pd.DataFrame({
        "open":  [100, 98],
        "high":  [105, 108],   # breaks up
        "low":   [95,  90],    # breaks down → outside
        "close": [102, 105],
        "volume": [1e6, 1e6],
    }, index=pd.date_range("2020-01-01", periods=2, freq="D"))
    out = classify_bars(df)
    assert out.iloc[1]["bar_type"] == TYPE_OUTSIDE


# ─── patterns ───────────────────────────────────────────────────────────────


def _make_bars(h, l, o=None, c=None):
    """Construct a DataFrame from high/low arrays. open=close=mid by default."""
    n = len(h)
    if o is None:
        o = [(h[i] + l[i]) / 2 for i in range(n)]
    if c is None:
        c = [(h[i] + l[i]) / 2 for i in range(n)]
    return pd.DataFrame({
        "open": o, "high": h, "low": l, "close": c, "volume": [1e6] * n,
    }, index=pd.date_range("2020-01-01", periods=n, freq="D"))


def test_pattern_2d_1_2u_bullish():
    """Down directional → inside → up directional = 2d-1-2u bullish reversal."""
    df = _make_bars(
        h=[100, 95, 92, 97, 103],     # bar3=92<95 (2d), bar4=97<97 no — wait
        l=[90,  85, 88, 90, 95],
    )
    # Reconstruct more carefully to match pattern on bars 2,3,4:
    # bar 2 (idx 2): down directional (breaks bar 1's low, doesn't break high)
    # bar 3 (idx 3): inside (within bar 2)
    # bar 4 (idx 4): up directional (breaks bar 3's high, not low)
    df = _make_bars(
        h=[110, 105, 100, 98,  108],
        l=[100, 95,  85,  88,  95],
    )
    # bar0: baseline
    # bar1 (105, 95): vs bar0 (110, 100): high < bar0, low < bar0 → 2d
    # bar2 (100, 85): vs bar1 (105, 95): high < bar1, low < bar1 → 2d
    # bar3 (98, 88): vs bar2 (100, 85): high < bar2, low > bar2 → inside
    # bar4 (108, 95): vs bar3 (98, 88): high > bar3, low > bar3 → 2u
    classified = classify_bars(df)
    detected = detect_patterns(classified)
    # Pattern fires on bar4
    assert detected.iloc[4]["strat_pattern"] == "2d-1-2u"
    assert detected.iloc[4]["strat_direction"] == "bullish"


def test_pattern_3_1_2u_high_confidence():
    """Outside → inside → up directional = 3-1-2u high-confidence bullish reversal."""
    df = _make_bars(
        h=[100, 105, 110, 107, 115],
        l=[90,  95,  85,  90,  100],
    )
    # bar0 baseline
    # bar1 (105, 95): vs bar0 (100,90): h>, l>  → 2u
    # bar2 (110, 85): vs bar1 (105, 95): h>, l< → outside (3)
    # bar3 (107, 90): vs bar2 (110, 85): h<, l> → inside (1)
    # bar4 (115, 100): vs bar3 (107, 90): h>, l> → 2u
    classified = classify_bars(df)
    detected = detect_patterns(classified)
    assert detected.iloc[4]["strat_pattern"] == "3-1-2u"
    assert detected.iloc[4]["strat_direction"] == "bullish"


def test_pattern_2d_2u_quick_reversal():
    """Down directional immediately followed by up directional (no inside) = 2d-2u."""
    df = _make_bars(
        h=[110, 105, 100, 108],
        l=[100, 95,  85,  98],
    )
    # bar0 baseline
    # bar1 (105, 95): vs bar0: h<, l< → 2d
    # bar2 (100, 85): vs bar1: h<, l< → 2d
    # bar3 (108, 98): vs bar2 (100, 85): h>, l> → 2u
    classified = classify_bars(df)
    detected = detect_patterns(classified)
    assert detected.iloc[3]["strat_pattern"] == "2d-2u"
    assert detected.iloc[3]["strat_direction"] == "bullish"


# ─── FTFC ───────────────────────────────────────────────────────────────────


def test_ftfc_all_bullish():
    """When all 4 higher-tf bars are bullish at a given trade-tf bar, FTFC is bullish."""
    trade_idx = pd.date_range("2020-01-06 09:00", periods=3, freq="h")
    trade_bars = pd.DataFrame({
        "open": [100, 101, 102], "high": [102, 103, 104],
        "low":  [99, 100, 101],  "close": [101, 102, 103],
        "volume": [1e6, 1e6, 1e6],
    }, index=trade_idx)

    # 4 higher-tf series, all bullish (close > open)
    higher = {}
    for name in ["1mo", "1w", "1d", "1h"]:
        higher[name] = pd.DataFrame({
            "open":  [90, 95],
            "high":  [100, 105],
            "low":   [85, 92],
            "close": [99, 104],  # bullish
            "volume": [1e6, 1e6],
        }, index=pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-06 08:00"]))

    ftfc = compute_ftfc(trade_bars, higher, threshold=0.75)
    assert ftfc["is_bullish_ftfc"].all()
    assert not ftfc["is_bearish_ftfc"].any()
    assert (ftfc["ftfc_bullish_score"] == 1.0).all()


def test_ftfc_mixed_below_threshold():
    trade_idx = pd.date_range("2020-01-06 09:00", periods=1, freq="h")
    trade_bars = pd.DataFrame({
        "open": [100], "high": [102], "low": [99], "close": [101], "volume": [1e6],
    }, index=trade_idx)

    higher = {
        "1mo": pd.DataFrame({"open": [100], "high": [110], "low": [95], "close": [108], "volume": [1e6]},
                            index=pd.DatetimeIndex(["2020-01-01 00:00"])),   # bullish
        "1w":  pd.DataFrame({"open": [100], "high": [110], "low": [95], "close": [108], "volume": [1e6]},
                            index=pd.DatetimeIndex(["2020-01-06 00:00"])),   # bullish
        "1d":  pd.DataFrame({"open": [100], "high": [110], "low": [95], "close": [95],  "volume": [1e6]},
                            index=pd.DatetimeIndex(["2020-01-06 00:00"])),   # bearish
        "1h":  pd.DataFrame({"open": [100], "high": [110], "low": [95], "close": [95],  "volume": [1e6]},
                            index=pd.DatetimeIndex(["2020-01-06 08:00"])),   # bearish
    }
    ftfc = compute_ftfc(trade_bars, higher, threshold=0.75)
    # 2 bullish of 4 → 0.5 < 0.75 → not bullish FTFC
    assert not ftfc["is_bullish_ftfc"].iloc[0]
    assert ftfc["ftfc_bullish_score"].iloc[0] == 0.5


# ─── simulator ──────────────────────────────────────────────────────────────


def test_simulate_2bar_bullish_levels_correct():
    """2d-2u bullish trade levels — entry=prior.high, stop=prior.low, target=pre_prior.high."""
    from strategy_engine.backtest.strat.simulator import _compute_trade_levels
    # bar 0: baseline (pre_prior for pattern)
    # bar 1: 2d (prior)
    # bar 2: 2u (signal)
    # Then bars 3,4 to have forward data
    df = _make_bars(
        h=[110, 105, 108, 115, 120],
        l=[100, 95,  98,  105, 112],
    )
    classified = classify_bars(df)
    detected = detect_patterns(classified)
    assert detected.iloc[2]["strat_pattern"] == "2d-2u"

    levels = _compute_trade_levels(detected, signal_idx=2, pattern="2d-2u", direction="bullish")
    assert levels is not None
    entry, target, stop = levels
    # Expected: entry=prior.high=105, stop=prior.low=95, target=pre_prior.high=110
    assert entry == 105.0, f"entry expected 105, got {entry}"
    assert stop == 95.0, f"stop expected 95, got {stop}"
    assert target == 110.0, f"target expected 110, got {target}"


def test_simulate_target_hit_bullish_3bar():
    """A clean 3-1-2u bullish with strong upside move → target hit."""
    # Construct: bars 0-1-2-3-4 form 3-1-2u at bar 4 (as above).
    # Then bars 5-6 rise strongly to hit target (prior bar high = 105).
    df = _make_bars(
        h=[100, 105, 110, 107, 115, 120, 125],
        l=[90,  95,  85,  90,  100, 110, 115],
    )
    classified = classify_bars(df)
    detected = detect_patterns(classified)
    # Force bar 4 to be a 3-1-2u signal
    assert detected.iloc[4]["strat_pattern"] == "3-1-2u"

    # No FTFC gating for this test
    params = StratParams(require_ftfc=False, min_risk_reward=0.1, max_holding_bars=5)
    trades, n_raw, n_ftfc = simulate_trades(detected, ftfc=None, params=params)
    assert n_raw >= 1
    assert len(trades) >= 1
    t = trades[0]
    assert t.direction == "bullish"
    assert t.pattern == "3-1-2u"
    # entry at inside-bar high (107), target at prior-bar (bar2) high (110), stop at inside-bar low (90)
    assert t.entry_price == 107.0
    assert t.target_price == 110.0
    assert t.stop_price == 90.0
    # Forward bar 5 has high 120 >= target 110 → target hit
    assert t.exit_reason == "target"
    assert t.pct_return > 0


def test_simulate_ftfc_filters_signals():
    """When FTFC is not aligned, signal is filtered out."""
    df = _make_bars(
        h=[100, 105, 110, 107, 115, 120, 125],
        l=[90,  95,  85,  90,  100, 110, 115],
    )
    classified = classify_bars(df)
    detected = detect_patterns(classified)

    # Build a bearish-FTFC frame on the trade index → bullish signal should be filtered
    ftfc = pd.DataFrame({
        "is_bullish_ftfc": [False] * len(detected),
        "is_bearish_ftfc": [True] * len(detected),
        "ftfc_bullish_score": [0.0] * len(detected),
        "ftfc_bearish_score": [1.0] * len(detected),
    }, index=detected.index)

    params = StratParams(require_ftfc=True, min_risk_reward=0.1, max_holding_bars=5)
    trades, n_raw, n_ftfc = simulate_trades(detected, ftfc=ftfc, params=params)
    assert n_raw >= 1
    assert n_ftfc == 0      # all signals filtered
    assert len(trades) == 0
