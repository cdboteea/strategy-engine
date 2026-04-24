"""
Transaction cost model for backtests.

v1 model — flat per-leg, symmetric for entry + exit:
  - `spread_bps`      half-spread paid on every leg (default 2 bps)
  - `slippage_bps`    adverse fill beyond mid (default 1 bps)
  - `commission_bps`  brokerage fee per leg (default 1 bps)

Total one-way cost = spread + slippage + commission
Round-trip cost   = 2 × one-way

Cost is applied by adjusting the pct_return of each trade:
    net_return = gross_return - 2 * (spread + slippage + commission) / 10_000

This means every trade loses ~8 bps round-trip under defaults. For a +5%
target that's 4.92 % net. For strategies that scalp on sub-1 % moves it's
catastrophic — which is the POINT: costless backtests flatter strategies
that can't actually survive real execution.

The model is attached to a backtest via `CostModel.from_strategy(strategy)`:
  - Reads `backtest.cost_model` block from YAML if present
  - Falls back to sensible defaults matching a retail-broker equity setup

Future extensions (out of scope for v1):
  - Volume-proportional slippage (larger trades move the market)
  - Asset-class defaults (crypto vs futures vs equity)
  - Borrow cost for short trades
  - Explicit bid-ask from market data
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CostModel:
    """Per-leg cost model. All values in basis points (1 bp = 0.01 %)."""
    spread_bps: float = 2.0
    slippage_bps: float = 1.0
    commission_bps: float = 1.0

    @classmethod
    def zero(cls) -> "CostModel":
        """Costless baseline — preserves legacy backtest behavior."""
        return cls(spread_bps=0.0, slippage_bps=0.0, commission_bps=0.0)

    @classmethod
    def retail_equity(cls) -> "CostModel":
        """Defaults for a US retail-broker equity strategy (~8 bps round trip)."""
        return cls(spread_bps=2.0, slippage_bps=1.0, commission_bps=1.0)

    @classmethod
    def institutional_equity(cls) -> "CostModel":
        """Institutional execution — tighter spread, higher slippage risk."""
        return cls(spread_bps=0.5, slippage_bps=2.0, commission_bps=0.1)

    @classmethod
    def from_strategy(cls, strategy: Any) -> "CostModel":
        """Extract a cost model from a Strategy object.

        YAML shape (optional top-level `cost_model` block):

            cost_model:
              profile: retail-equity   # or 'zero', 'institutional-equity'
              # OR explicit overrides:
              spread_bps: 3.0
              slippage_bps: 1.5
              commission_bps: 0.5

        If no block is present, returns `retail_equity()`. If `profile` is
        set, uses that profile as the base; any explicit bps overrides
        that profile per field.
        """
        block = getattr(strategy, "cost_model", None)
        if block is None:
            return cls.retail_equity()
        # Pydantic model or dict
        if hasattr(block, "model_dump"):
            data = block.model_dump()
        elif isinstance(block, dict):
            data = dict(block)
        else:
            data = {}

        profile = data.get("profile")
        if profile == "zero":
            base = cls.zero()
        elif profile == "institutional-equity":
            base = cls.institutional_equity()
        else:
            base = cls.retail_equity()

        # Per-field override: explicit non-None in `data` wins; else profile default.
        # (Pydantic's model_dump() includes None for unset optional fields, so we
        # must check for None explicitly rather than relying on dict.get default.)
        def _pick(field: str, default: float) -> float:
            v = data.get(field)
            return float(v) if v is not None else float(default)

        return cls(
            spread_bps=_pick("spread_bps", base.spread_bps),
            slippage_bps=_pick("slippage_bps", base.slippage_bps),
            commission_bps=_pick("commission_bps", base.commission_bps),
        )

    @classmethod
    def by_name(cls, name: str) -> "CostModel":
        """Look up a named profile by string — for CLI --cost-profile flag."""
        if name == "zero":
            return cls.zero()
        if name == "retail-equity":
            return cls.retail_equity()
        if name == "institutional-equity":
            return cls.institutional_equity()
        raise ValueError(
            f"unknown cost profile {name!r}. Supported: zero, retail-equity, institutional-equity"
        )

    @classmethod
    def flat_round_trip(cls, round_trip_bps: float) -> "CostModel":
        """Create a flat cost model from a target round-trip cost in bps.

        The target is split as (spread=bps/2, slippage=bps/4, commission=bps/4)
        per leg — proportions chosen to preserve the retail-equity ratio
        (spread dominates, slippage + commission are secondary). Useful for
        cost-sensitivity sweeps (e.g. `--round-trip-bps 20`).
        """
        if round_trip_bps < 0:
            raise ValueError(f"round_trip_bps must be >= 0, got {round_trip_bps}")
        if round_trip_bps == 0:
            return cls.zero()
        # Per leg = round_trip / 2; spread:slippage:commission = 2:1:1 per leg
        per_leg = round_trip_bps / 2.0
        return cls(
            spread_bps=per_leg * 0.5,
            slippage_bps=per_leg * 0.25,
            commission_bps=per_leg * 0.25,
        )

    # ── Cost computation ────────────────────────────────────────────────

    @property
    def one_way_bps(self) -> float:
        """Total cost for one leg (entry OR exit), in bps."""
        return self.spread_bps + self.slippage_bps + self.commission_bps

    @property
    def round_trip_bps(self) -> float:
        """Total cost for a round-trip trade (entry + exit), in bps."""
        return 2 * self.one_way_bps

    @property
    def round_trip_pct(self) -> float:
        """Round-trip cost as a pct return decimal (e.g. 0.0008 = 0.08 %)."""
        return self.round_trip_bps / 10_000.0

    def apply_to_return(self, gross_pct_return: float) -> float:
        """Net a gross pct return by the round-trip cost."""
        return gross_pct_return - self.round_trip_pct

    def __str__(self) -> str:
        return (
            f"CostModel(spread={self.spread_bps}bp, slippage={self.slippage_bps}bp, "
            f"commission={self.commission_bps}bp, round_trip={self.round_trip_bps}bp)"
        )
