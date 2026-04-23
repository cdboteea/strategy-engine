"""
Promotion gates — decide whether a strategy can advance from `backtested` → `promoted`.

v1 gates (stress test deferred to v2 — needs regime-specific run infrastructure):

  G1 — OOS Sharpe ≥ threshold           (quality)
  G2 — OOS max drawdown ≥ -threshold    (risk)
  G3 — Min total OOS trades              (statistical significance)
  G4 — Walk-forward stability            (OOS Sharpe std across folds)
  G5 — Minimum win rate                  (hit-rate floor)

NOTE on Sharpe semantics: all Sharpes here are **equity-curve Sharpes** —
portfolio daily-return Sharpes annualized by sqrt(bars_per_year), not per-trade
return Sharpes. For infrequent weekly strategies with cash idle time, the
bar for equity Sharpe is lower than for trade-Sharpe. Calibrate thresholds
per strategy family accordingly.

v2 additions (future):
  - Stress test (2008, 2020 regimes): DD ≤ 2× baseline
  - Cost sensitivity: Sharpe at 5 bp slippage ≥ 0.7 of baseline
  - Regime consistency: positive Sharpe in bull AND flat regimes
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from ..backtest.walkforward import WalkForwardResult


# ─── PORTFOLIO profile (default) ───────────────────────────────────────────
# For strategies that compete for portfolio allocation. Judged on full-bar
# equity Sharpe (includes idle-bar noise). Designed for Bollinger-like
# medium-frequency strategies.
DEFAULT_THRESHOLDS = {
    "min_active_sharpe": 0.6,       # full-bar Sharpe, mean over active folds
    "max_drawdown": -0.30,
    "min_trades": 10,
    "max_active_sharpe_std": 0.75,
    "min_active_win_rate": 0.55,
    "min_activation_rate": 0.25,
}

# ─── ACTIVE-TRADER profile ─────────────────────────────────────────────────
# For strategies used to trade actively on specific setups (STRAT-family).
# Judged on active-bar Sharpe — "quality when invested" — because idle-bar
# noise swamps their full-bar Sharpe at typical 1-3% activation rates.
# Expects: many trades, strong per-active-bar edge, modest activation is OK.
ACTIVE_TRADER_THRESHOLDS = {
    "min_active_bar_sharpe": 1.5,    # mean active-bar Sharpe across active folds
    "max_drawdown": -0.15,           # tighter than portfolio; max-DD IS the risk guard
    "min_trades": 30,
    "min_active_win_rate": 0.55,
    "min_activation_rate": 0.50,
}
# Note: no cross-fold stability gate for active-trader. Active-bar Sharpes on
# tiny samples (1-3% of bars) swing wildly fold-to-fold and don't give a
# reliable stability signal. max_drawdown handles the risk side instead.

# Profile registry
PROFILES = {
    "portfolio": DEFAULT_THRESHOLDS,
    "active-trader": ACTIVE_TRADER_THRESHOLDS,
}


@dataclass
class GateResult:
    name: str
    passed: bool
    value: float | int | None
    threshold: float | int | None
    reason: str = ""


@dataclass
class PromotionDecision:
    strategy_id: str
    passed: bool
    gates: list[GateResult] = field(default_factory=list)
    summary: str = ""


def check_gates(
    strategy_id: str,
    wf: WalkForwardResult,
    thresholds: Optional[dict] = None,
    profile: str = "portfolio",
) -> PromotionDecision:
    """Apply gates against a walk-forward result.

    `profile` selects the preset threshold dictionary:
      - 'portfolio' (default): judges on full-bar equity Sharpe
      - 'active-trader': judges on active-bar Sharpe (for STRAT-family)
    `thresholds` (if provided) override the profile's defaults.
    """
    base = PROFILES.get(profile, DEFAULT_THRESHOLDS)
    t = {**base, **(thresholds or {})}
    gates: list[GateResult] = []

    if profile == "active-trader":
        return _check_active_trader_gates(strategy_id, wf, t)

    return _check_portfolio_gates(strategy_id, wf, t)


def _check_portfolio_gates(
    strategy_id: str,
    wf: WalkForwardResult,
    t: dict,
) -> PromotionDecision:
    gates: list[GateResult] = []

    # G1 — Active-fold mean Sharpe
    g1_pass = wf.oos_active_mean_sharpe >= t["min_active_sharpe"]
    gates.append(GateResult(
        name="min_active_sharpe",
        passed=g1_pass,
        value=round(wf.oos_active_mean_sharpe, 3),
        threshold=t["min_active_sharpe"],
        reason=(
            f"Active-fold mean Sharpe {wf.oos_active_mean_sharpe:.3f} "
            f"{'≥' if g1_pass else '<'} {t['min_active_sharpe']} "
            f"(n_active={wf.n_active_folds}/{wf.n_folds})"
        ),
    ))

    # G2 — Max drawdown
    g2_pass = wf.oos_worst_dd >= t["max_drawdown"]
    gates.append(GateResult(
        name="max_drawdown",
        passed=g2_pass,
        value=round(wf.oos_worst_dd, 4),
        threshold=t["max_drawdown"],
        reason=(
            f"Worst OOS DD {wf.oos_worst_dd:.2%} "
            f"{'≥' if g2_pass else '<'} {t['max_drawdown']:.0%}"
        ),
    ))

    # G3 — Min trades
    g3_pass = wf.oos_total_trades >= t["min_trades"]
    gates.append(GateResult(
        name="min_trades",
        passed=g3_pass,
        value=wf.oos_total_trades,
        threshold=t["min_trades"],
        reason=(
            f"Total OOS trades {wf.oos_total_trades} "
            f"{'≥' if g3_pass else '<'} {t['min_trades']}"
        ),
    ))

    # G4 — Active-fold Sharpe stability
    g4_pass = wf.oos_active_std_sharpe <= t["max_active_sharpe_std"]
    gates.append(GateResult(
        name="active_sharpe_stability",
        passed=g4_pass,
        value=round(wf.oos_active_std_sharpe, 3),
        threshold=t["max_active_sharpe_std"],
        reason=(
            f"Active-fold Sharpe std {wf.oos_active_std_sharpe:.3f} "
            f"{'≤' if g4_pass else '>'} {t['max_active_sharpe_std']}"
        ),
    ))

    # G5 — Active-fold mean win rate
    g5_pass = wf.oos_active_mean_win_rate >= t["min_active_win_rate"]
    gates.append(GateResult(
        name="min_active_win_rate",
        passed=g5_pass,
        value=round(wf.oos_active_mean_win_rate, 3),
        threshold=t["min_active_win_rate"],
        reason=(
            f"Mean OOS win rate (active folds) {wf.oos_active_mean_win_rate:.1%} "
            f"{'≥' if g5_pass else '<'} {t['min_active_win_rate']:.0%}"
        ),
    ))

    # G6 — Activation rate (how often does the strategy fire?)
    g6_pass = wf.activation_rate >= t["min_activation_rate"]
    gates.append(GateResult(
        name="min_activation_rate",
        passed=g6_pass,
        value=round(wf.activation_rate, 3),
        threshold=t["min_activation_rate"],
        reason=(
            f"Activation rate {wf.activation_rate:.1%} "
            f"{'≥' if g6_pass else '<'} {t['min_activation_rate']:.0%} "
            f"({wf.n_active_folds}/{wf.n_folds} folds fired)"
        ),
    ))

    return _finalize(strategy_id, gates)


def _check_active_trader_gates(
    strategy_id: str,
    wf: WalkForwardResult,
    t: dict,
) -> PromotionDecision:
    gates: list[GateResult] = []

    # AT-1 — active-bar Sharpe (the primary signal)
    g1_pass = wf.oos_active_bar_mean_sharpe >= t["min_active_bar_sharpe"]
    gates.append(GateResult(
        name="min_active_bar_sharpe",
        passed=g1_pass,
        value=round(wf.oos_active_bar_mean_sharpe, 3),
        threshold=t["min_active_bar_sharpe"],
        reason=(
            f"Active-bar mean Sharpe {wf.oos_active_bar_mean_sharpe:.3f} "
            f"{'≥' if g1_pass else '<'} {t['min_active_bar_sharpe']} "
            f"(active folds {wf.n_active_folds}/{wf.n_folds}, "
            f"mean bar coverage {wf.oos_mean_active_bar_fraction:.1%})"
        ),
    ))

    # AT-2 — tighter DD
    g2_pass = wf.oos_worst_dd >= t["max_drawdown"]
    gates.append(GateResult(
        name="max_drawdown",
        passed=g2_pass,
        value=round(wf.oos_worst_dd, 4),
        threshold=t["max_drawdown"],
        reason=(
            f"Worst OOS DD {wf.oos_worst_dd:.2%} "
            f"{'≥' if g2_pass else '<'} {t['max_drawdown']:.0%}"
        ),
    ))

    # AT-3 — more trades
    g3_pass = wf.oos_total_trades >= t["min_trades"]
    gates.append(GateResult(
        name="min_trades",
        passed=g3_pass,
        value=wf.oos_total_trades,
        threshold=t["min_trades"],
        reason=(
            f"Total OOS trades {wf.oos_total_trades} "
            f"{'≥' if g3_pass else '<'} {t['min_trades']}"
        ),
    ))

    # AT-4 — win rate
    g4_pass = wf.oos_active_mean_win_rate >= t["min_active_win_rate"]
    gates.append(GateResult(
        name="min_active_win_rate",
        passed=g4_pass,
        value=round(wf.oos_active_mean_win_rate, 3),
        threshold=t["min_active_win_rate"],
        reason=(
            f"Mean OOS win rate (active folds) {wf.oos_active_mean_win_rate:.1%} "
            f"{'≥' if g4_pass else '<'} {t['min_active_win_rate']:.0%}"
        ),
    ))

    # AT-5 — activation
    g5_pass = wf.activation_rate >= t["min_activation_rate"]
    gates.append(GateResult(
        name="min_activation_rate",
        passed=g5_pass,
        value=round(wf.activation_rate, 3),
        threshold=t["min_activation_rate"],
        reason=(
            f"Activation rate {wf.activation_rate:.1%} "
            f"{'≥' if g5_pass else '<'} {t['min_activation_rate']:.0%} "
            f"({wf.n_active_folds}/{wf.n_folds} folds fired)"
        ),
    ))

    return _finalize(strategy_id, gates)


def _finalize(strategy_id: str, gates: list[GateResult]) -> PromotionDecision:
    all_pass = all(g.passed for g in gates)
    n_pass = sum(1 for g in gates if g.passed)
    n_total = len(gates)
    summary = (
        f"{n_pass}/{n_total} gates passed" +
        ("" if all_pass else " — strategy NOT promoted.")
    )
    return PromotionDecision(
        strategy_id=strategy_id,
        passed=all_pass,
        gates=gates,
        summary=summary,
    )
