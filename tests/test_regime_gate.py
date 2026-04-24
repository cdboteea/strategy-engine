"""Behavioral tests for the regime-gate primitive (Stage B3)."""
from __future__ import annotations
import pandas as pd
import pytest

from strategy_engine.backtest.regime import (
    VixGate,
    GateApplicationStats,
    apply_vix_gate_to_signals,
    gate_from_config,
)
from strategy_engine.registry.schema import Strategy


# ── VixGate unit tests ─────────────────────────────────────────────────────

def test_vix_gate_below_default_threshold():
    gate = VixGate(mode="below", threshold=35.0)
    assert gate.evaluate(15.0) is True         # calm → fire
    assert gate.evaluate(34.99) is True
    assert gate.evaluate(35.0) is False        # strict
    assert gate.evaluate(80.0) is False        # panic → suppress


def test_vix_gate_above():
    gate = VixGate(mode="above", threshold=20.0)
    assert gate.evaluate(15.0) is False
    assert gate.evaluate(20.0) is False        # strict
    assert gate.evaluate(25.0) is True


def test_vix_gate_between():
    gate = VixGate(mode="between", lower=15.0, upper=30.0)
    assert gate.evaluate(10.0) is False
    assert gate.evaluate(15.0) is True         # inclusive
    assert gate.evaluate(20.0) is True
    assert gate.evaluate(30.0) is True
    assert gate.evaluate(30.01) is False


def test_vix_gate_requires_threshold_for_below_above():
    with pytest.raises(ValueError, match="threshold"):
        VixGate(mode="below")
    with pytest.raises(ValueError, match="threshold"):
        VixGate(mode="above")


def test_vix_gate_between_requires_both_bounds():
    with pytest.raises(ValueError, match="lower AND upper"):
        VixGate(mode="between", lower=10.0)


def test_vix_gate_describe_readable():
    assert "VIX < 35" in VixGate(mode="below", threshold=35.0).describe()
    assert "VIX > 20" in VixGate(mode="above", threshold=20.0).describe()
    assert "15.0 <= VIX <= 30.0" in VixGate(
        mode="between", lower=15.0, upper=30.0,
    ).describe()


# ── apply_vix_gate_to_signals behavior ──────────────────────────────────

def _make_vix_bars(index: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": closes}, index=pd.to_datetime(index))


def test_apply_gate_drops_panic_signals():
    gate = VixGate(mode="below", threshold=35.0)
    # VIX history: mostly low, one panic spike
    vix = _make_vix_bars(
        ["2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26", "2026-02-02"],
        [15.0, 18.0, 42.0, 50.0, 20.0],
    )
    # Signals on weeks 2 (VIX 18) and 4 (VIX 50)
    signals = [pd.Timestamp("2026-01-12"), pd.Timestamp("2026-01-26")]

    kept, stats = apply_vix_gate_to_signals(signals, gate, vix_bars=vix)

    assert len(kept) == 1
    assert kept[0] == pd.Timestamp("2026-01-12")
    assert stats.n_signals_in == 2
    assert stats.n_signals_out == 1
    assert stats.n_dropped == 1


def test_apply_gate_keeps_all_when_vix_always_calm():
    gate = VixGate(mode="below", threshold=35.0)
    vix = _make_vix_bars(
        ["2026-01-05", "2026-01-12", "2026-01-19"],
        [12.0, 15.0, 18.0],
    )
    signals = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-12"),
                pd.Timestamp("2026-01-19")]

    kept, stats = apply_vix_gate_to_signals(signals, gate, vix_bars=vix)

    assert len(kept) == 3
    assert stats.n_dropped == 0


def test_apply_gate_drops_all_when_vix_always_panicked():
    gate = VixGate(mode="below", threshold=35.0)
    vix = _make_vix_bars(
        ["2026-01-05", "2026-01-12"],
        [45.0, 50.0],
    )
    signals = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-12")]

    kept, stats = apply_vix_gate_to_signals(signals, gate, vix_bars=vix)
    assert kept == []
    assert stats.n_dropped == 2


def test_apply_gate_uses_most_recent_vix_on_or_before_signal():
    """If signal is on a Wednesday and VIX bar is the prior Friday, we use
    that prior close."""
    gate = VixGate(mode="below", threshold=35.0)
    # VIX bars: Mon Jan 5 = 15 (calm), Mon Jan 12 = 50 (panic)
    vix = _make_vix_bars(["2026-01-05", "2026-01-12"], [15.0, 50.0])
    # Signal on Wed Jan 7 — should use Jan 5's VIX (calm → pass)
    signals_wed = [pd.Timestamp("2026-01-07")]
    kept, stats = apply_vix_gate_to_signals(signals_wed, gate, vix_bars=vix)
    assert len(kept) == 1
    assert stats.n_dropped == 0

    # Signal on Wed Jan 14 — should use Jan 12's VIX (panic → drop)
    signals_wed2 = [pd.Timestamp("2026-01-14")]
    kept2, stats2 = apply_vix_gate_to_signals(signals_wed2, gate, vix_bars=vix)
    assert kept2 == []
    assert stats2.n_dropped == 1


def test_apply_gate_keeps_signal_when_no_vix_data():
    """Pre-VIX-history signals should KEEP (fail-open, not fail-closed)."""
    gate = VixGate(mode="below", threshold=35.0)
    vix = _make_vix_bars(["2026-01-05"], [15.0])
    # Signal BEFORE any VIX data we have
    signals = [pd.Timestamp("2025-01-01")]
    kept, stats = apply_vix_gate_to_signals(signals, gate, vix_bars=vix)
    assert len(kept) == 1                      # conservative: kept
    assert stats.n_no_vix_data == 1


def test_apply_gate_empty_vix_data_keeps_everything():
    gate = VixGate(mode="below", threshold=35.0)
    signals = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-12")]
    kept, stats = apply_vix_gate_to_signals(signals, gate, vix_bars=pd.DataFrame())
    assert len(kept) == 2
    assert stats.n_no_vix_data == 2


# ── gate_from_config ─────────────────────────────────────────────────────

def test_gate_from_config_below():
    cfg = {"type": "vix", "mode": "below", "threshold": 35.0}
    g = gate_from_config(cfg)
    assert isinstance(g, VixGate)
    assert g.mode == "below"
    assert g.threshold == 35.0


def test_gate_from_config_between():
    cfg = {"type": "vix", "mode": "between", "lower": 15.0, "upper": 30.0}
    g = gate_from_config(cfg)
    assert g.lower == 15.0 and g.upper == 30.0


def test_gate_from_config_none_returns_none():
    assert gate_from_config(None) is None
    assert gate_from_config({}) is None


def test_gate_from_config_unsupported_type():
    with pytest.raises(ValueError, match="not supported"):
        gate_from_config({"type": "breadth", "threshold": 0.5})


# ── Schema integration ────────────────────────────────────────────────

def _strategy_yaml_base() -> dict:
    return {
        "id": "test-regime", "name": "Test", "status": "draft",
        "asset_class": "equity-index", "instruments": ["SPY"], "timeframe": "1w",
        "signal_logic": {"type": "bollinger-mean-reversion"},
        "entry": {"mode": "hybrid-50-50"},
        "exit": {"mode": "profit-target", "target": 0.05, "forward_window_weeks": 13},
        "capital_allocation": 0.1,
        "data_sources": ["firstrate"],
    }


def test_schema_accepts_regime_gate_below():
    y = _strategy_yaml_base()
    y["regime_gate"] = {"type": "vix", "mode": "below", "threshold": 35}
    s = Strategy.model_validate(y)
    assert s.regime_gate is not None
    assert s.regime_gate.mode == "below"
    assert s.regime_gate.threshold == 35.0


def test_schema_rejects_between_without_bounds():
    y = _strategy_yaml_base()
    y["regime_gate"] = {"type": "vix", "mode": "between", "lower": 15}
    with pytest.raises(Exception):
        Strategy.model_validate(y)


def test_schema_rejects_below_without_threshold():
    y = _strategy_yaml_base()
    y["regime_gate"] = {"type": "vix", "mode": "below"}
    with pytest.raises(Exception):
        Strategy.model_validate(y)


def test_schema_rejects_between_with_lower_gt_upper():
    y = _strategy_yaml_base()
    y["regime_gate"] = {"type": "vix", "mode": "between", "lower": 30, "upper": 15}
    with pytest.raises(Exception, match="lower.*upper"):
        Strategy.model_validate(y)
