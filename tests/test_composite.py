"""Behavioral tests for the CompositeStrategy framework.

These tests exercise the filter logic directly with synthetic SignalEvents
and synthetic primary-trade objects — they do NOT run the full backtest
pipeline. End-to-end integration is covered by the registry-level spec test.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
import pandas as pd
import pytest

from strategy_engine.backtest.composite import (
    SignalEvent,
    filter_primary_trades,
    _matches_within_window,
)
from strategy_engine.registry.schema import Strategy


# ---- _matches_within_window unit tests ------------------------------------

def _ev(day: str, direction: str = "bullish", src: str = "x") -> SignalEvent:
    return SignalEvent(
        date=pd.Timestamp(day),
        direction=direction,
        source_id=src,
        source_type="strat-pattern",
    )


def test_window_hit_within_range():
    primary = _ev("2024-06-10")
    confs = [_ev("2024-06-07")]  # 3 days before
    assert _matches_within_window(primary, confs, window_days=5, require_direction_match=True)


def test_window_miss_outside_range():
    primary = _ev("2024-06-10")
    confs = [_ev("2024-05-20")]  # 21 days before
    assert not _matches_within_window(primary, confs, window_days=14, require_direction_match=True)


def test_direction_mismatch_blocks_match():
    primary = _ev("2024-06-10", "bullish")
    confs = [_ev("2024-06-09", "bearish")]
    assert not _matches_within_window(primary, confs, window_days=14, require_direction_match=True)
    # With require_direction_match=False, bearish near-date is still a hit:
    assert _matches_within_window(primary, confs, window_days=14, require_direction_match=False)


def test_window_zero_requires_exact_date():
    primary = _ev("2024-06-10")
    confs_same = [_ev("2024-06-10")]
    confs_next = [_ev("2024-06-11")]
    assert _matches_within_window(primary, confs_same, window_days=0, require_direction_match=True)
    assert not _matches_within_window(primary, confs_next, window_days=0, require_direction_match=True)


# ---- filter_primary_trades behavior ---------------------------------------

@dataclass
class _FakeTrade:
    signal_date: pd.Timestamp


def test_mode_any_keeps_trade_with_single_source_hit():
    primary_trades = [_FakeTrade(pd.Timestamp("2024-06-10"))]
    confirmations = {
        "src-a": [_ev("2024-06-09")],
        "src-b": [],
    }
    kept, stats = filter_primary_trades(
        primary_trades, confirmations, mode="any",
        window_days=5, require_direction_match=True,
    )
    assert len(kept) == 1
    assert stats.primary_trades_raw == 1
    assert stats.primary_trades_after_confirmation == 1


def test_mode_all_drops_when_one_source_misses():
    primary_trades = [_FakeTrade(pd.Timestamp("2024-06-10"))]
    confirmations = {
        "src-a": [_ev("2024-06-09")],  # hits
        "src-b": [_ev("2024-05-01")],  # too far
    }
    kept, stats = filter_primary_trades(
        primary_trades, confirmations, mode="all",
        window_days=5, require_direction_match=True,
    )
    assert kept == []
    assert stats.primary_trades_after_confirmation == 0
    assert stats.drop_reason_counts.get("missing-one-or-more-confirmations") == 1


def test_mode_all_keeps_trade_when_every_source_hits():
    primary_trades = [_FakeTrade(pd.Timestamp("2024-06-10"))]
    confirmations = {
        "src-a": [_ev("2024-06-09")],
        "src-b": [_ev("2024-06-11")],
    }
    kept, _ = filter_primary_trades(
        primary_trades, confirmations, mode="all",
        window_days=5, require_direction_match=True,
    )
    assert len(kept) == 1


def test_no_confirmation_events_drops_all_any_mode():
    primary_trades = [_FakeTrade(pd.Timestamp("2024-06-10"))]
    confirmations = {"src-a": []}
    kept, stats = filter_primary_trades(
        primary_trades, confirmations, mode="any",
        window_days=14, require_direction_match=True,
    )
    assert kept == []
    assert stats.drop_reason_counts["no-confirmation-in-window"] == 1


def test_filter_stats_track_confirmation_counts():
    primary_trades = [_FakeTrade(pd.Timestamp("2024-06-10"))]
    confirmations = {
        "src-a": [_ev("2024-06-09"), _ev("2024-08-01")],  # 2 events
        "src-b": [_ev("2024-06-11")],                      # 1 event
    }
    _, stats = filter_primary_trades(
        primary_trades, confirmations, mode="any",
        window_days=5, require_direction_match=True,
    )
    assert stats.confirmations_per_source == {"src-a": 2, "src-b": 1}


# ---- Schema validation ----------------------------------------------------

def _base_composite_yaml() -> dict:
    return {
        "id": "test-composite",
        "name": "Test composite",
        "status": "draft",
        "asset_class": "equity-index",
        "instruments": ["SPY"],
        "timeframe": "1w",
        "signal_logic": {"type": "composite"},
        "entry": {"mode": "inherit-from-primary"},
        "exit": {"mode": "inherit-from-primary"},
        "capital_allocation": 0.1,
        "data_sources": ["firstrate"],
        "composite": {
            "primary": "spy-bollinger-hybrid-v1",
            "confirmations": ["strat-2d-2u-1d-spy-v1"],
            "mode": "any",
            "window_days": 14,
            "require_direction_match": True,
        },
    }


def test_composite_schema_accepts_valid_yaml():
    s = Strategy.model_validate(_base_composite_yaml())
    assert s.composite is not None
    assert s.composite.mode == "any"
    assert s.composite.window_days == 14
    assert s.composite.confirmations == ["strat-2d-2u-1d-spy-v1"]


def test_composite_type_without_block_rejected():
    y = _base_composite_yaml()
    y.pop("composite")
    with pytest.raises(Exception) as exc:
        Strategy.model_validate(y)
    assert "composite" in str(exc.value).lower()


def test_composite_block_without_type_rejected():
    y = _base_composite_yaml()
    y["signal_logic"] = {"type": "bollinger-mean-reversion"}
    with pytest.raises(Exception) as exc:
        Strategy.model_validate(y)
    assert "composite" in str(exc.value).lower()


def test_composite_mode_must_be_any_or_all():
    y = _base_composite_yaml()
    y["composite"]["mode"] = "majority"
    with pytest.raises(Exception):
        Strategy.model_validate(y)


def test_composite_requires_at_least_one_confirmation():
    y = _base_composite_yaml()
    y["composite"]["confirmations"] = []
    with pytest.raises(Exception):
        Strategy.model_validate(y)


def test_composite_window_days_upper_bound():
    y = _base_composite_yaml()
    y["composite"]["window_days"] = 1000
    with pytest.raises(Exception):
        Strategy.model_validate(y)
