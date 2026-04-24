"""Behavioral tests for the transaction-cost model + its integration with
Bollinger + STRAT + Composite backtests (Stage B1)."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import pytest

from strategy_engine.backtest.costs import CostModel
from strategy_engine.backtest import bollinger as bol
from strategy_engine.registry.schema import Strategy


# ── CostModel unit tests ────────────────────────────────────────────────────

def test_cost_model_defaults_retail_equity():
    cm = CostModel()
    assert cm.spread_bps == 2.0
    assert cm.slippage_bps == 1.0
    assert cm.commission_bps == 1.0
    assert cm.one_way_bps == 4.0
    assert cm.round_trip_bps == 8.0
    assert cm.round_trip_pct == pytest.approx(0.0008)


def test_cost_model_zero_profile():
    cm = CostModel.zero()
    assert cm.round_trip_bps == 0.0
    assert cm.apply_to_return(0.05) == 0.05  # no cost


def test_cost_model_institutional_profile():
    cm = CostModel.institutional_equity()
    # Institutional: 0.5 + 2.0 + 0.1 = 2.6 per leg, 5.2 round trip
    assert cm.round_trip_bps == pytest.approx(5.2)


def test_cost_model_apply_to_return_nets_cost():
    cm = CostModel.retail_equity()
    # +5% gross → +5% - 0.08% = +4.92%
    assert cm.apply_to_return(0.05) == pytest.approx(0.0492, abs=1e-6)
    # -5% gross → worse: -5.08%
    assert cm.apply_to_return(-0.05) == pytest.approx(-0.0508, abs=1e-6)


def test_cost_model_by_name_resolves_profiles():
    assert CostModel.by_name("zero").round_trip_bps == 0
    assert CostModel.by_name("retail-equity").round_trip_bps == 8.0
    assert CostModel.by_name("institutional-equity").round_trip_bps == pytest.approx(5.2)
    with pytest.raises(ValueError, match="unknown cost profile"):
        CostModel.by_name("bogus")


def test_cost_model_from_strategy_without_block_defaults_to_retail():
    @dataclass
    class _Strat:
        cost_model = None
    cm = CostModel.from_strategy(_Strat())
    assert cm.round_trip_bps == 8.0


def test_cost_model_from_strategy_dict_overrides_profile_fields():
    class _Block:
        def model_dump(self):
            return {"profile": "institutional-equity", "spread_bps": 10.0}
    class _Strat:
        def __init__(self): self.cost_model = _Block()
    cm = CostModel.from_strategy(_Strat())
    # Starts from institutional (0.5, 2.0, 0.1) then overrides spread → 10.0
    assert cm.spread_bps == 10.0
    assert cm.slippage_bps == 2.0
    assert cm.commission_bps == pytest.approx(0.1)
    assert cm.one_way_bps == pytest.approx(12.1)


def test_cost_model_from_strategy_zero_profile():
    class _Block:
        def model_dump(self):
            return {"profile": "zero"}
    class _Strat:
        def __init__(self): self.cost_model = _Block()
    assert CostModel.from_strategy(_Strat()).round_trip_bps == 0


# ── Schema — cost_model block ──────────────────────────────────────────────

def _base_bollinger_yaml() -> dict:
    return {
        "id": "test-bol-cost", "name": "Test", "status": "draft",
        "asset_class": "equity-index", "instruments": ["SPY"], "timeframe": "1w",
        "signal_logic": {"type": "bollinger-mean-reversion"},
        "entry": {"mode": "hybrid-50-50"},
        "exit": {"mode": "profit-target", "target": 0.05, "forward_window_weeks": 13},
        "capital_allocation": 0.1, "data_sources": ["firstrate"],
    }


def test_schema_accepts_cost_model_block():
    y = _base_bollinger_yaml()
    y["cost_model"] = {"profile": "institutional-equity", "spread_bps": 0.3}
    s = Strategy.model_validate(y)
    cm = CostModel.from_strategy(s)
    assert cm.spread_bps == pytest.approx(0.3)


def test_schema_rejects_unknown_profile():
    y = _base_bollinger_yaml()
    y["cost_model"] = {"profile": "yolo"}
    with pytest.raises(Exception):
        Strategy.model_validate(y)


def test_schema_rejects_negative_bps():
    y = _base_bollinger_yaml()
    y["cost_model"] = {"spread_bps": -1.0}
    with pytest.raises(Exception):
        Strategy.model_validate(y)


# ── Integration: costs reduce pct_return on trades ──────────────────────────

def _build_bars_with_dip_and_recovery() -> pd.DataFrame:
    """Synthetic series that guarantees at least one completable Bollinger trade.

    Structure:
      - bars 0..19   flat around 100  (builds the 20-bar lookback)
      - bar 20       dips to 92       (breaks lower band → signal fires)
      - bars 21..25  stays around 93-94 (hybrid 2nd-half trigger hits at -5%)
      - bars 26..50  recovers past the +5% target
    This provides enough forward bars (min 11 required) for the trade to
    be simulated to completion.
    """
    import numpy as np
    rng = pd.date_range("2026-01-01", periods=50, freq="D")
    seed = np.random.default_rng(42)
    base = np.full(50, 100.0) + seed.normal(0, 0.2, 50)
    # Force a dip early + gradual recovery
    base[20] = 92.0
    for i in range(21, 26):
        base[i] = 93.0 + seed.normal(0, 0.2)
    for i in range(26, 50):
        # Linear recovery to 108 (well above +5% target of ~96.6)
        base[i] = 93.0 + (i - 26) * 0.6
    return pd.DataFrame({
        "open": base - 0.2, "high": base + 0.4, "low": base - 0.4,
        "close": base, "volume": 1_000_000,
    }, index=rng)


def test_bollinger_with_zero_cost_matches_legacy_behavior():
    bars = _build_bars_with_dip_and_recovery()
    params = bol.BollingerParams(lookback=20, std_dev=2.0, profit_target=0.05)
    r_no_cost = bol.run_bollinger(bars, params, capital_allocation=0.1,
                                   timeframe="1d", cost_model=CostModel.zero())
    r_default = bol.run_bollinger(bars, params, capital_allocation=0.1,
                                   timeframe="1d")  # no cost_model param
    # Both should produce identical trades (no cost applied)
    assert r_no_cost.win_rate == r_default.win_rate
    assert r_no_cost.total_pnl_pct == pytest.approx(r_default.total_pnl_pct, abs=1e-9)


def test_bollinger_with_retail_cost_reduces_pct_returns():
    bars = _build_bars_with_dip_and_recovery()
    params = bol.BollingerParams(lookback=20, std_dev=2.0, profit_target=0.05)

    r_zero = bol.run_bollinger(bars, params, capital_allocation=0.1,
                                 timeframe="1d", cost_model=CostModel.zero())
    r_retail = bol.run_bollinger(bars, params, capital_allocation=0.1,
                                   timeframe="1d", cost_model=CostModel.retail_equity())

    # We should have at least one trade
    assert r_zero.n_trades >= 1
    assert r_retail.n_trades == r_zero.n_trades

    # Each trade's pct_return should be lower by exactly round_trip_pct (0.0008)
    round_trip = CostModel.retail_equity().round_trip_pct
    for t_zero, t_retail in zip(r_zero.trades, r_retail.trades):
        assert t_retail.pct_return == pytest.approx(
            t_zero.pct_return - round_trip, abs=1e-9,
        )

    # And aggregate total_pnl_pct must reflect the cost
    assert r_retail.total_pnl_pct < r_zero.total_pnl_pct


def test_cost_flows_through_to_equity_sharpe():
    """Regression: cost model MUST reduce equity_sharpe as bps increases.
    Prior to this test, cost only hit per-trade pct_return; equity curve
    was built from raw bar returns and ignored cost entirely, so
    equity_sharpe was identical at 0 vs 20 bps. This test locks in the fix
    (cost deducted from strategy_return at each trade's exit bar)."""
    bars = _build_bars_with_dip_and_recovery()
    params = bol.BollingerParams(lookback=20, std_dev=2.0, profit_target=0.05)

    # 0 bps baseline
    r_zero = bol.run_bollinger(bars, params, capital_allocation=0.1,
                                 timeframe="1d", cost_model=CostModel.zero())
    # 8 bps retail
    r_retail = bol.run_bollinger(bars, params, capital_allocation=0.1,
                                   timeframe="1d", cost_model=CostModel.retail_equity())
    # 20 bps heavy
    r_heavy = bol.run_bollinger(bars, params, capital_allocation=0.1,
                                  timeframe="1d", cost_model=CostModel.flat_round_trip(20))

    # Must have the same trade count
    assert r_zero.n_trades == r_retail.n_trades == r_heavy.n_trades
    assert r_zero.n_trades >= 1

    # Equity-curve metrics MUST differ (strictly decreasing as cost rises)
    assert r_zero.equity_sharpe > r_retail.equity_sharpe > r_heavy.equity_sharpe, (
        f"Sharpe must decrease with cost: zero={r_zero.equity_sharpe:.4f}, "
        f"retail={r_retail.equity_sharpe:.4f}, heavy={r_heavy.equity_sharpe:.4f}"
    )
    assert r_zero.equity_total_return > r_retail.equity_total_return > r_heavy.equity_total_return


def test_flat_round_trip_preserves_retail_ratios():
    """flat_round_trip(8) should produce 8bp round-trip with the expected
    spread:slippage:commission split (2:1:1 per leg)."""
    cm = CostModel.flat_round_trip(8)
    assert cm.round_trip_bps == pytest.approx(8.0)
    # Per leg = 4 bps; split as 2:1:1 → spread=2, slippage=1, commission=1
    assert cm.spread_bps == pytest.approx(2.0)
    assert cm.slippage_bps == pytest.approx(1.0)
    assert cm.commission_bps == pytest.approx(1.0)


def test_flat_round_trip_zero_returns_zero_model():
    cm = CostModel.flat_round_trip(0)
    assert cm.round_trip_bps == 0.0
    assert cm.spread_bps == 0.0


def test_flat_round_trip_rejects_negative():
    with pytest.raises(ValueError, match=">= 0"):
        CostModel.flat_round_trip(-5)


def test_heavy_cost_flips_winner_to_loser_on_actual_trade():
    """A small winner (+0.5%) becomes a loser once 2% round-trip cost applies."""
    from strategy_engine.backtest.bollinger import BollingerTrade
    ts = pd.Timestamp("2026-04-01")
    trades = [BollingerTrade(
        signal_date=ts, signal_price=100.0, first_half_entry_price=100.0,
        second_half_entry_price=None, second_half_entry_date=None,
        avg_entry_price=100.0, exit_date=ts + pd.Timedelta(days=5),
        exit_price=100.5, exit_reason="target", holding_bars=5,
        pct_return=0.005, target_hit=True, trough_pct=0.0,
    )]
    heavy = CostModel(spread_bps=50.0, slippage_bps=50.0, commission_bps=0.0)
    # round_trip = 200 bps = 2%; summarize (called WITHOUT bars) skips equity curve
    r = bol.summarize(trades, bars=None, capital_allocation=0.1,
                      timeframe="1d", cost_model=heavy)
    # Net return = 0.005 - 0.02 = -0.015
    assert trades[0].pct_return == pytest.approx(-0.015, abs=1e-6)
    assert r.win_rate == 0.0  # former winner is now a loser
