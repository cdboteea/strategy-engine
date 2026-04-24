"""Pydantic schema for registry YAML entries.

Every strategy YAML in ~/clawd/research/strategies/**/*.yaml must validate
against the `Strategy` model. Use `strategy-engine registry validate` to check.
"""
from __future__ import annotations
from datetime import date
from typing import Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from ..config import (
    STATUSES,
    ASSET_CLASSES,
    TIMEFRAMES,
    SIGNAL_TYPES,
    DATA_SOURCES,
)


class BacktestWindow(BaseModel):
    """
    Optional date window for backtests. When unset, engine runs on full history.
    Useful for:
      - Reproducing earlier research that used a specific window
      - Excluding a regime the strategy wasn't designed for
      - Forward-test hold-out (set end date earlier than today, then re-run)

    CLI flags `--start` / `--end` override the YAML window at run time.
    """
    model_config = ConfigDict(extra="forbid")

    start: Optional[date] = None
    end: Optional[date] = None

    @model_validator(mode="after")
    def _check_order(self) -> "BacktestWindow":
        if self.start and self.end and self.start > self.end:
            raise ValueError(f"backtest_window.start ({self.start}) > end ({self.end})")
        return self


class PromotionLastCheck(BaseModel):
    """Latest promotion-gate check, whether it passed or failed."""
    model_config = ConfigDict(extra="allow")

    date: str
    passed: bool
    failed_gates: list[str] = Field(default_factory=list)


class PromotionMeta(BaseModel):
    """Promotion metadata, written by `strategy-engine promote`."""
    model_config = ConfigDict(extra="allow")

    decision_date: Optional[str] = None     # set when strategy is promoted
    last_check: Optional[PromotionLastCheck] = None  # set when a check fails


class SignalLogic(BaseModel):
    """Signal definition. Type-specific fields held in `params`."""
    model_config = ConfigDict(extra="allow")

    type: str
    # Type-specific params live as sibling keys; we accept any via `extra='allow'`
    # and validate inside `type` branches if needed.

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in SIGNAL_TYPES:
            raise ValueError(f"signal_logic.type must be one of {SIGNAL_TYPES}, got {v!r}")
        return v


class RegimeGateBlock(BaseModel):
    """Optional regime-gate filter — suspends signal firing when broad-market
    indicator (e.g. VIX) is outside the acceptable range.

    Typical Bollinger usage: `{type: vix, mode: below, threshold: 35}` — skip
    mean-reversion entries when VIX > 35 (panic-vol regime where the
    13-week forward window may not capture the bounce).

    Only evaluated at the primary signal-detection step. Once a trade is
    open, it runs to completion under normal exit rules.
    """
    model_config = ConfigDict(extra="forbid")

    type: Literal["vix"] = "vix"
    mode: Literal["below", "above", "between"] = "below"
    threshold: Optional[float] = Field(default=None, ge=0)
    lower: Optional[float] = Field(default=None, ge=0)
    upper: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_mode_fields(self) -> "RegimeGateBlock":
        if self.mode in ("below", "above") and self.threshold is None:
            raise ValueError(f"regime_gate mode='{self.mode}' requires `threshold`")
        if self.mode == "between":
            if self.lower is None or self.upper is None:
                raise ValueError("regime_gate mode='between' requires `lower` AND `upper`")
            if self.lower > self.upper:
                raise ValueError(f"regime_gate: lower ({self.lower}) > upper ({self.upper})")
        return self


class CostModelBlock(BaseModel):
    """Per-strategy transaction cost model override.

    Uses `profile` (named preset) OR explicit bps fields. When both
    are set, explicit fields override the profile's defaults per field.

    Defaults when this block is absent: `profile: retail-equity`
    (2 bp spread + 1 bp slippage + 1 bp commission per leg).
    """
    model_config = ConfigDict(extra="forbid")

    profile: Optional[Literal["zero", "retail-equity", "institutional-equity"]] = None
    spread_bps: Optional[float] = Field(default=None, ge=0)
    slippage_bps: Optional[float] = Field(default=None, ge=0)
    commission_bps: Optional[float] = Field(default=None, ge=0)


class CompositeMeta(BaseModel):
    """
    Composite-strategy wiring. Only used when `signal_logic.type == 'composite'`.

    A composite runs its `primary` strategy end-to-end, then keeps only those
    primary signals whose date has a matching `confirmations` signal within
    ±window_days. This lets Bollinger signals be filtered by STRAT pattern
    confirmation (or any combination of registered strategies).

    v1 constraint: `primary` must reference a `bollinger-mean-reversion`
    strategy. Confirmations may be any signal type. Entry/exit logic is
    inherited from the primary.
    """
    model_config = ConfigDict(extra="forbid")

    primary: str = Field(..., min_length=3)
    confirmations: list[str] = Field(..., min_length=1)
    mode: Literal["any", "all"] = "any"
    window_days: int = Field(default=3, ge=0, le=365)
    require_direction_match: bool = True


class Entry(BaseModel):
    """Entry rules. Mode-specific fields held loosely."""
    model_config = ConfigDict(extra="allow")

    mode: str


class Exit(BaseModel):
    """Exit rules. Mode-specific fields held loosely."""
    model_config = ConfigDict(extra="allow")

    mode: str
    stop_loss: Optional[Any] = None
    target: Optional[Any] = None


class Strategy(BaseModel):
    """Full strategy record."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=3, max_length=120)
    name: str = Field(..., min_length=3, max_length=200)
    status: str
    asset_class: str
    instruments: list[str] = Field(..., min_length=1)
    timeframe: str
    signal_logic: SignalLogic
    entry: Entry
    exit: Exit
    capital_allocation: float = Field(..., ge=0, le=1)
    data_sources: list[str] = Field(..., min_length=1)
    backtest_window: Optional[BacktestWindow] = None
    backtest_runs: list[str] = Field(default_factory=list)
    live_run: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    notes_doc: Optional[str] = None
    promotion: Optional[PromotionMeta] = None
    composite: Optional[CompositeMeta] = None
    cost_model: Optional[CostModelBlock] = None
    regime_gate: Optional[RegimeGateBlock] = None
    retired_at: Optional[str] = None
    retirement_reason: Optional[str] = None

    @model_validator(mode="after")
    def _check_composite(self) -> "Strategy":
        # If signal_logic.type is 'composite', a composite block is required.
        if self.signal_logic.type == "composite" and self.composite is None:
            raise ValueError(
                "signal_logic.type='composite' requires a top-level `composite:` block"
            )
        if self.composite is not None and self.signal_logic.type != "composite":
            raise ValueError(
                f"`composite:` block present but signal_logic.type is "
                f"{self.signal_logic.type!r} (must be 'composite')"
            )
        return self

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        if v not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        return v

    @field_validator("asset_class")
    @classmethod
    def _validate_asset_class(cls, v: str) -> str:
        if v not in ASSET_CLASSES:
            raise ValueError(f"asset_class must be one of {ASSET_CLASSES}")
        return v

    @field_validator("timeframe")
    @classmethod
    def _validate_timeframe(cls, v: str) -> str:
        if v not in TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {TIMEFRAMES}")
        return v

    @field_validator("data_sources")
    @classmethod
    def _validate_data_sources(cls, vs: list[str]) -> list[str]:
        bad = [v for v in vs if v not in DATA_SOURCES]
        if bad:
            raise ValueError(f"unknown data_sources: {bad}. Allowed: {DATA_SOURCES}")
        return vs

    @field_validator("id")
    @classmethod
    def _validate_id_slug(cls, v: str) -> str:
        import re
        if not re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$", v):
            raise ValueError(
                "id must be kebab-case (lowercase alphanumeric + hyphens, "
                "no leading/trailing hyphens)"
            )
        return v
