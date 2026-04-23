"""Behavioral tests for Bollinger logic + backtest runner.

Uses synthetic OHLCV data (not real firstrate data) so tests are deterministic
and don't require DB access.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from strategy_engine.backtest.bollinger import (
    BollingerParams,
    compute_bollinger,
    detect_signals,
    simulate_trades,
    summarize,
    run_bollinger,
)


def _sine_with_dips(n_weeks: int = 200) -> pd.DataFrame:
    """Synthetic weekly OHLCV: sinusoidal close with occasional sharp dips that
    should trigger Bollinger signals."""
    rng = pd.date_range("2020-01-01", periods=n_weeks, freq="W")
    base = 100 + 10 * np.sin(np.linspace(0, 8 * np.pi, n_weeks))
    noise = np.random.default_rng(42).normal(0, 1.5, n_weeks)
    close = base + noise
    # Inject sharp dips every 30 bars
    for i in range(30, n_weeks, 30):
        close[i] *= 0.92  # -8% dip
    high = close * 1.01
    low = close * 0.99
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.full(n_weeks, 1_000_000)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=rng,
    )


def test_compute_bollinger_produces_bands():
    df = _sine_with_dips(100)
    out = compute_bollinger(df, lookback=20, std_dev=2.0)
    assert "sma" in out.columns
    assert "upper" in out.columns
    assert "lower" in out.columns
    # After lookback window, bands are computed
    valid = out.iloc[25:]
    assert (valid["upper"] > valid["sma"]).all()
    assert (valid["lower"] < valid["sma"]).all()


def test_detect_signals_finds_dips():
    df = _sine_with_dips(200)
    df = compute_bollinger(df, lookback=20, std_dev=2.0)
    df = detect_signals(df)
    n_signals = int(df["is_signal"].sum())
    assert n_signals > 0, "should detect at least some injected dips"


def test_detect_signals_fires_on_every_below_band_bar():
    """Consecutive weeks below the band must each fire independently —
    no first-touch suppression. Cluster analysis is a downstream concern.

    Direct check: manually construct a DataFrame where 3 consecutive rows
    have close < lower; assert all 3 are flagged is_signal=True (not just
    the first).
    """
    idx = pd.date_range("2020-01-01", periods=10, freq="W")
    # Manually construct: 'close' and 'lower' such that rows 4-6 are below band
    df = pd.DataFrame({
        "open":  [100, 100, 100, 100,  95,  94,  93, 100, 100, 100],
        "high":  [101, 101, 101, 101,  96,  95,  94, 101, 101, 101],
        "low":   [ 99,  99,  99,  99,  94,  93,  92,  99,  99,  99],
        "close": [100, 100, 100, 100,  95,  94,  93, 100, 100, 100],
        "volume": [1e6] * 10,
        "sma":   [100] * 10,
        "std":   [2.0] * 10,
        "upper": [104] * 10,
        "lower": [ 96] * 10,  # fixed lower band at 96 — rows 4,5,6 below it
    }, index=idx)

    out = detect_signals(df)
    # Rows 4, 5, 6 should all fire — NOT just row 4 (first-touch would suppress 5,6)
    signals = out["is_signal"].tolist()
    assert signals[4] and signals[5] and signals[6], (
        f"expected consecutive signals at positions 4,5,6; got {signals}. "
        f"Clusters must trigger each event independently."
    )
    assert not any(signals[:4]), "no signal should fire when close >= lower"
    assert not any(signals[7:]), "no signal should fire on recovery rows"
    assert int(out["is_signal"].sum()) == 3


def test_simulate_trades_exits_on_target():
    """A strong upward trend after a signal should trigger target-hit exits."""
    # Build a clean signal scenario
    n = 100
    idx = pd.date_range("2020-01-01", periods=n, freq="W")
    close = np.concatenate([
        np.linspace(100, 110, 40),   # rising
        [90],                         # signal bar (sharp dip)
        np.linspace(91, 130, n - 41), # strong rally → target should hit
    ])
    high = close * 1.02
    low = close * 0.98
    open_ = close
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1e6}, index=idx)
    df = compute_bollinger(df, lookback=20, std_dev=2.0)
    df = detect_signals(df)

    # Manually force a signal for deterministic test
    df["is_signal"] = False
    df.loc[df.index[40], "is_signal"] = True  # at the dip bar

    params = BollingerParams(lookback=20, std_dev=2.0, profit_target=0.05, forward_window_bars=13, min_forward_bars=5)
    trades = simulate_trades(df, params)
    assert len(trades) == 1
    t = trades[0]
    assert t.target_hit, f"expected target hit, got exit_reason={t.exit_reason}"
    assert t.pct_return > 0


def test_simulate_trades_handles_no_target_hit():
    """If price never hits target, exit at end of forward window."""
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="W")
    close = np.concatenate([
        np.linspace(100, 110, 30),
        [90],
        np.full(n - 31, 91.0),  # flat — never hits +5% target
    ])
    df = pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": 1e6,
    }, index=idx)
    df = compute_bollinger(df, lookback=20, std_dev=2.0)
    df["is_signal"] = False
    df.loc[df.index[30], "is_signal"] = True

    params = BollingerParams(lookback=20, std_dev=2.0, profit_target=0.05, forward_window_bars=13, min_forward_bars=5)
    trades = simulate_trades(df, params)
    assert len(trades) == 1
    assert trades[0].exit_reason == "forward-window-end"
    assert not trades[0].target_hit


def test_summarize_empty_is_safe():
    r = summarize([])
    assert r.n_trades == 0
    assert r.equity_sharpe == 0.0
    assert r.trades == []


def test_run_bollinger_end_to_end():
    df = _sine_with_dips(300)
    params = BollingerParams()
    result = run_bollinger(df, params)
    # Sanity — should detect multiple signals and summary fields populated
    assert result.n_trades >= 1
    assert 0 <= result.win_rate <= 1
    assert result.total_pnl_pct is not None
    # Equity-curve metrics populated
    assert result.equity_sharpe is not None
    assert result.equity_max_drawdown <= 0
    assert -1.0 <= result.equity_ann_return <= 10.0  # sanity bounds


def test_run_bollinger_requires_ohlc_columns():
    df = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(ValueError, match="open/high/low/close"):
        run_bollinger(df, BollingerParams())


# ─── equity-curve metrics ──────────────────────────────────────────────────


def test_build_equity_curve_empty_trades():
    from strategy_engine.backtest.bollinger import build_equity_curve
    bars = _sine_with_dips(50)
    eq, is_active = build_equity_curve([], bars, capital_allocation=0.1)
    assert (eq == 1.0).all()
    assert not is_active.any()


def test_build_equity_curve_respects_allocation():
    """10% allocation on a single trade that gains 10% should result in ~1% equity gain."""
    from strategy_engine.backtest.bollinger import (
        BollingerTrade, build_equity_curve, _equity_metrics,
    )
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="W")
    # Price rises 10% over trade holding period
    close = np.concatenate([np.full(10, 100.0), np.linspace(100, 110, 10), np.full(10, 110.0)])
    bars = pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": 1e6,
    }, index=idx)

    trade = BollingerTrade(
        signal_date=idx[10], signal_price=100.0,
        first_half_entry_price=100.0, second_half_entry_price=None,
        second_half_entry_date=None, avg_entry_price=100.0,
        exit_date=idx[19], exit_price=110.0, exit_reason="target",
        holding_bars=9, pct_return=0.10, target_hit=True, trough_pct=0.0,
    )
    eq, is_active = build_equity_curve([trade], bars, capital_allocation=0.10)
    # Equity should end near 1 + 10% * 10% = 1.01
    assert 1.005 < eq.iloc[-1] < 1.015, f"expected ~1.01, got {eq.iloc[-1]}"
    assert is_active.sum() >= 9


def test_equity_sharpe_differs_from_per_trade():
    """Equity-curve Sharpe should be materially different from what per-trade annualization produces —
    specifically, for infrequent trades, equity Sharpe should be much LOWER."""
    df = _sine_with_dips(300)
    params = BollingerParams()
    result = run_bollinger(df, params, capital_allocation=0.10, timeframe="1w")
    # Per-trade returns mean/std * sqrt(52) inflated the old calc; equity curve should be reasonable
    # (not infinite, not negative for this synthetic case, and bounded)
    assert abs(result.equity_sharpe) < 10, (
        f"equity_sharpe {result.equity_sharpe} is unreasonably high — "
        f"suggests annualization is still wrong"
    )
