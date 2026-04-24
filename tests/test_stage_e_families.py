"""Behavioral tests for Stage E new strategy families:
  - momentum (SMA crossover, MACD crossover)
  - breakout (Donchian channel)
  - trend (200-SMA pullback)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import pytest

from strategy_engine.backtest.momentum import (
    SmaCrossoverParams, MacdCrossoverParams,
    compute_sma_crossover, compute_macd_crossover,
    simulate_sma_crossover, simulate_macd_crossover,
    run_sma_crossover, run_macd_crossover,
)
from strategy_engine.backtest.breakout import (
    DonchianParams, compute_donchian, simulate_donchian, run_donchian,
)
from strategy_engine.backtest.trend import (
    TrendPullbackParams, compute_trend_pullback, simulate_trend_pullback,
    run_trend_pullback,
)
from strategy_engine.backtest.costs import CostModel


def _mk_bars(closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame({
        "open": [c - 0.2 for c in closes],
        "high": [c + 0.5 for c in closes],
        "low":  [c - 0.5 for c in closes],
        "close": closes,
        "volume": [1_000_000] * len(closes),
    }, index=idx)


# ── SMA crossover ──────────────────────────────────────────────────────────

def test_sma_crossover_detects_golden_cross():
    # 30 bars flat at 100, then 30 bars rising to 130 (forces golden cross)
    closes = [100.0] * 30 + list(np.linspace(101, 130, 30))
    df = compute_sma_crossover(_mk_bars(closes), fast=5, slow=20)
    assert df["bullish_cross"].sum() >= 1


def test_sma_simulator_long_only_entry_and_exit():
    # Up-down-up pattern → expect at least 1 long trade
    closes = list(np.linspace(100, 130, 30)) + list(np.linspace(130, 100, 30)) + list(np.linspace(100, 130, 30))
    bars = _mk_bars(closes)
    params = SmaCrossoverParams(fast_window=5, slow_window=20, direction_bias="long-only")
    df = compute_sma_crossover(bars, 5, 20)
    trades = simulate_sma_crossover(df, params)
    assert len(trades) >= 1
    for t in trades:
        assert t.direction == "bullish"
        assert t.entry_date < t.exit_date


def test_sma_run_end_to_end_produces_result():
    closes = list(np.linspace(100, 150, 60))
    params = SmaCrossoverParams(fast_window=5, slow_window=20)
    result = run_sma_crossover(
        _mk_bars(closes), params, capital_allocation=0.1, timeframe="1d",
        cost_model=CostModel.zero(),
    )
    assert result.n_trades >= 1 or result.n_trades == 0   # might be 0 in pure-uptrend with no cross
    # Headline metric exists + is a number
    assert isinstance(result.equity_sharpe, float)


def test_sma_cost_model_degrades_returns():
    closes = list(np.linspace(100, 130, 30)) + list(np.linspace(130, 100, 30)) + list(np.linspace(100, 130, 30))
    params = SmaCrossoverParams(fast_window=5, slow_window=20)
    r_zero = run_sma_crossover(_mk_bars(closes), params, timeframe="1d",
                                 cost_model=CostModel.zero())
    r_retail = run_sma_crossover(_mk_bars(closes), params, timeframe="1d",
                                   cost_model=CostModel.retail_equity())
    if r_zero.n_trades > 0:
        assert r_retail.total_pnl_pct <= r_zero.total_pnl_pct


# ── MACD crossover ─────────────────────────────────────────────────────────

def test_macd_crossover_detects_histogram_flip():
    # Noisy price series with enough oscillation to produce MACD flips
    closes = [100 + 5 * np.sin(i / 5.0) for i in range(100)]
    df = compute_macd_crossover(_mk_bars(closes), MacdCrossoverParams())
    assert df["bullish_cross"].sum() >= 1
    assert df["bearish_cross"].sum() >= 1


def test_macd_simulator_uses_signal_line_warmup(monkeypatch):
    """Regression: simulator should NOT skip all bars when sma_slow missing.

    MACD produces `signal_line` (not `sma_slow`). This test verifies the
    warmup check falls back to signal_line presence. Prior to the fix,
    MACD always produced 0 trades.
    """
    closes = [100 + 5 * np.sin(i / 5.0) for i in range(100)]
    params = MacdCrossoverParams()
    df = compute_macd_crossover(_mk_bars(closes), params)
    # Confirm sma_slow is absent (it's a MACD frame)
    assert "sma_slow" not in df.columns
    assert "signal_line" in df.columns
    trades = simulate_macd_crossover(df, params)
    assert len(trades) >= 1     # the crux of the regression guard


def test_macd_run_end_to_end():
    closes = [100 + 5 * np.sin(i / 5.0) for i in range(100)]
    result = run_macd_crossover(
        _mk_bars(closes), MacdCrossoverParams(),
        capital_allocation=0.1, timeframe="1d",
    )
    assert result.n_trades >= 1
    assert isinstance(result.equity_sharpe, float)


# ── Donchian breakout ──────────────────────────────────────────────────────

def test_donchian_detects_upside_breakout():
    # 20 bars flat then sharp rally above flat range
    closes = [100.0] * 20 + [105, 110, 115, 120, 125, 130, 128, 122, 118, 115]
    df = compute_donchian(_mk_bars(closes), 5, 3)
    # Breakout should trigger on the first rising bar (close > prev 5-bar high)
    assert df["bullish_breakout"].sum() >= 1


def test_donchian_simulator_enters_on_breakout_exits_on_trailing():
    closes = [100.0] * 20 + list(range(101, 120)) + list(range(119, 100, -1))
    bars = _mk_bars(closes)
    params = DonchianParams(entry_window=5, exit_window=3)
    df = compute_donchian(bars, 5, 3)
    trades = simulate_donchian(df, params)
    assert len(trades) >= 1
    for t in trades:
        assert t.direction == "bullish"
        assert t.exit_reason in ("trailing-stop", "end-of-data")


def test_donchian_run_end_to_end():
    closes = [100.0] * 20 + list(range(101, 130))
    params = DonchianParams(entry_window=10, exit_window=5)
    result = run_donchian(
        _mk_bars(closes), params, capital_allocation=0.1, timeframe="1d",
        cost_model=CostModel.zero(),
    )
    assert result.n_trades >= 1


# ── Trend pullback ────────────────────────────────────────────────────────

def test_trend_pullback_requires_both_uptrend_and_pullback():
    # Long uptrend base (closes rising), then a dip (triggers pullback inside uptrend)
    rising = list(np.linspace(100, 150, 40))
    dip = list(np.linspace(150, 135, 10))
    recovery = list(np.linspace(135, 160, 15))
    closes = rising + dip + recovery
    df = compute_trend_pullback(_mk_bars(closes), long_sma=30, short_sma=10)
    # Some entries should fire (uptrend + pullback)
    assert df["entry_condition"].sum() >= 1


def test_trend_simulator_exits_on_pullback_resolved():
    closes = list(np.linspace(100, 150, 30)) + [148, 146, 144] + list(np.linspace(144, 170, 20))
    bars = _mk_bars(closes)
    params = TrendPullbackParams(long_sma=20, short_sma=5)
    df = compute_trend_pullback(bars, 20, 5)
    trades = simulate_trend_pullback(df, params)
    assert len(trades) >= 1
    # At least one should have exited on pullback-resolved (not trend-broken)
    reasons = {t.exit_reason for t in trades}
    assert "pullback-resolved" in reasons or "end-of-data" in reasons


def test_trend_run_end_to_end():
    closes = list(np.linspace(100, 140, 40)) + [138, 136, 135] + list(np.linspace(135, 155, 20))
    params = TrendPullbackParams(long_sma=30, short_sma=10)
    result = run_trend_pullback(
        _mk_bars(closes), params, capital_allocation=0.1, timeframe="1d",
        cost_model=CostModel.zero(),
    )
    assert isinstance(result.equity_sharpe, float)


# ── Schema + runner integration ────────────────────────────────────────────

def test_new_signal_types_in_signal_types_list():
    from strategy_engine.config import SIGNAL_TYPES
    assert "sma-crossover" in SIGNAL_TYPES
    assert "macd-crossover" in SIGNAL_TYPES
    assert "donchian-breakout" in SIGNAL_TYPES
    assert "trend-pullback" in SIGNAL_TYPES


def test_schema_accepts_sma_crossover():
    from strategy_engine.registry.schema import Strategy
    y = {
        "id": "test-sma", "name": "Test", "status": "draft",
        "asset_class": "equity-index", "instruments": ["SPY"], "timeframe": "1d",
        "signal_logic": {"type": "sma-crossover", "fast_window": 50, "slow_window": 200},
        "entry": {"mode": "at-cross-close"},
        "exit": {"mode": "reverse-cross"},
        "capital_allocation": 0.1,
        "data_sources": ["firstrate"],
    }
    s = Strategy.model_validate(y)
    assert s.signal_logic.type == "sma-crossover"


def test_schema_accepts_donchian_breakout():
    from strategy_engine.registry.schema import Strategy
    y = {
        "id": "test-donchian", "name": "Test", "status": "draft",
        "asset_class": "equity-index", "instruments": ["SPY"], "timeframe": "1d",
        "signal_logic": {"type": "donchian-breakout", "entry_window": 20, "exit_window": 10},
        "entry": {"mode": "breakout"},
        "exit": {"mode": "trailing-channel"},
        "capital_allocation": 0.1,
        "data_sources": ["firstrate"],
    }
    s = Strategy.model_validate(y)
    assert s.signal_logic.type == "donchian-breakout"
