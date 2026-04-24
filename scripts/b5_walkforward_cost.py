#!/usr/bin/env python3
"""
B5 — Walk-forward cost sensitivity.

Unlike B2 (single-backtest metrics), B5 re-runs walk-forward at 0 / 8 bps and
produces the authoritative gate comparison. The promotion gates apply to
walk-forward aggregates, so this is what actually matters for "would this
strategy still have been promoted under realistic costs?"

Run:
    ~/clawd/venv/bin/python scripts/b5_walkforward_cost.py \
        --output ~/clawd/docs/b5-wf-cost-2026-04-24.md
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from strategy_engine.backtest.costs import CostModel
from strategy_engine.backtest.walkforward import run_walkforward
from strategy_engine.backtest.runner import _find_yaml_path, _resolve_window
from strategy_engine.registry.loader import load_one, validate_all
from strategy_engine.providers.duckdb_provider import (
    load_ohlcv, load_multi_timeframe, DataNotAvailable,
)


# Gate thresholds — mirror of promotion/gates.py
PROFILE_GATES = {
    "portfolio": {
        "min_active_mean_sharpe": 0.60,
        "max_drawdown": 0.30,
        "min_trades": 20,
        "min_win_rate": 0.55,
        "min_activation": 0.25,
    },
    "active-trader": {
        "min_active_bar_mean_sharpe": 1.50,
        "max_drawdown": 0.15,
        "min_trades": 10,
        "min_win_rate": 0.50,
        "min_activation": 0.50,
    },
}


def _guess_profile(strategy) -> str:
    tags = [t.lower() for t in (strategy.tags or [])]
    if "active-trader" in tags or any("strat" in t for t in tags):
        return "active-trader"
    return "portfolio"


def _check_wf_gates(wf, profile: str) -> tuple[bool, list[str]]:
    """Check if a walk-forward result passes the profile's gates."""
    gates = PROFILE_GATES[profile]
    failed: list[str] = []

    if profile == "active-trader":
        if wf.oos_active_bar_mean_sharpe < gates["min_active_bar_mean_sharpe"]:
            failed.append(f"active_bar_sharpe<{gates['min_active_bar_mean_sharpe']}")
    else:
        if wf.oos_active_mean_sharpe < gates["min_active_mean_sharpe"]:
            failed.append(f"active_mean_sharpe<{gates['min_active_mean_sharpe']}")

    if abs(wf.oos_worst_dd) > gates["max_drawdown"]:
        failed.append(f"worst_dd>{gates['max_drawdown']}")

    if wf.oos_total_trades < gates["min_trades"]:
        failed.append(f"trades<{gates['min_trades']}")

    if wf.oos_active_mean_win_rate < gates["min_win_rate"]:
        failed.append(f"win_rate<{gates['min_win_rate']}")

    if wf.activation_rate < gates["min_activation"]:
        failed.append(f"activation<{gates['min_activation']}")

    return (not failed, failed)


def run_one(strategy_id: str, cost_bps: float):
    yaml_path = _find_yaml_path(strategy_id)
    if not yaml_path:
        return None, f"no YAML for {strategy_id}"
    strat = load_one(yaml_path)
    cost = CostModel.flat_round_trip(cost_bps)

    eff_start, eff_end = _resolve_window(strat, None, None)
    try:
        bars = load_ohlcv(strat.instruments[0], strat.timeframe,
                           start=eff_start, end=eff_end)
    except DataNotAvailable as e:
        return None, f"data unavailable: {e}"

    higher_tfs = None
    if strat.signal_logic.type == "strat-pattern":
        sl = strat.signal_logic
        if bool(getattr(sl, "require_ftfc", True)):
            ftfc_tfs = list(getattr(sl, "ftfc_timeframes", ["1mo", "1w", "1d", "1h"]))
            all_tfs = list(dict.fromkeys([*ftfc_tfs, strat.timeframe]))
            higher_tfs = load_multi_timeframe(
                strat.instruments[0], all_tfs, start=eff_start, end=eff_end,
            )

    wf = run_walkforward(
        strat, bars,
        train_years=3, test_years=1, step_years=1,
        higher_timeframes=higher_tfs,
        cost_model=cost,
    )
    return wf, None


def render(results: list[dict]) -> str:
    from collections import defaultdict
    by = defaultdict(dict)
    for r in results:
        by[r["strategy_id"]][r["bps"]] = r

    lines: list[str] = []
    lines.append("# B5 — Walk-Forward Cost Sensitivity")
    lines.append("")
    lines.append("**Date:** 2026-04-24")
    lines.append("**Scope:** every promoted strategy re-run through walk-forward cross-validation at 0 bps and 8 bps (retail-equity) round-trip cost. This is the authoritative gate check — B2's single-backtest proxy was misleading for sparse strategies.")
    lines.append("")
    lines.append("## 1. Verdict matrix (real walk-forward gates)")
    lines.append("")
    lines.append("| Strategy | Profile | 0 bps | 8 bps | Verdict |")
    lines.append("|---|---|---:|---:|---|")
    for sid in sorted(by):
        row = by[sid]
        r0 = row.get(0); r8 = row.get(8)
        if r0 is None or r8 is None:
            continue
        profile = r0["profile"]
        verdict_map = {True: "✅", False: "❌"}
        v0 = verdict_map[r0["passed"]]
        v8 = verdict_map[r8["passed"]]
        if r0["passed"] and r8["passed"]:
            note = "robust"
        elif r0["passed"] and not r8["passed"]:
            note = "**borderline — fails at real costs**"
        elif not r0["passed"] and r8["passed"]:
            note = "unusual (investigate)"
        else:
            note = "fails both tiers"
        lines.append(f"| `{sid}` | {profile} | {v0} | {v8} | {note} |")

    lines.append("")
    lines.append("## 2. Metric comparison")
    lines.append("")
    for sid in sorted(by):
        row = by[sid]
        if 0 not in row or 8 not in row:
            continue
        r0 = row[0]; r8 = row[8]
        profile = r0["profile"]
        lines.append(f"### `{sid}` — {profile}")
        lines.append("")
        m0 = r0["metrics"]; m8 = r8["metrics"]
        # Pick headline Sharpe per profile
        sharpe_key = "oos_active_bar_mean_sharpe" if profile == "active-trader" else "oos_active_mean_sharpe"
        lines.append(f"| Metric | 0 bps | 8 bps | Δ |")
        lines.append(f"|---|---:|---:|---:|")
        for label, key in [
            (f"Active {'bar' if profile=='active-trader' else 'fold'} Sharpe", sharpe_key),
            ("Active mean win rate", "oos_active_mean_win_rate"),
            ("Activation rate", "activation_rate"),
            ("Total trades", "oos_total_trades"),
            ("Worst fold DD", "oos_worst_dd"),
        ]:
            v0 = m0.get(key, 0); v8 = m8.get(key, 0)
            fmt = "{:.3f}" if isinstance(v0, float) and abs(v0) < 100 else "{}"
            try:
                delta = v8 - v0
                delta_str = f"{delta:+.3f}" if isinstance(v0, float) else f"{delta:+d}"
            except TypeError:
                delta_str = "—"
            lines.append(f"| {label} | {fmt.format(v0)} | {fmt.format(v8)} | {delta_str} |")
        verdict0 = "PASS" if r0["passed"] else f"FAIL ({','.join(r0['failed_gates'])})"
        verdict8 = "PASS" if r8["passed"] else f"FAIL ({','.join(r8['failed_gates'])})"
        lines.append(f"| **Verdict** | {verdict0} | {verdict8} | — |")
        lines.append("")

    lines.append("## 3. Recommendations")
    lines.append("")
    # Identify strategies that pass 0bps but fail 8bps
    borderline = []
    robust = []
    broken = []
    for sid in sorted(by):
        row = by[sid]
        if 0 in row and 8 in row:
            if row[0]["passed"] and not row[8]["passed"]:
                borderline.append(sid)
            elif row[0]["passed"] and row[8]["passed"]:
                robust.append(sid)
            elif not row[0]["passed"] and not row[8]["passed"]:
                broken.append(sid)
    if robust:
        lines.append("### ✅ Robust at real costs — keep as promoted")
        for sid in robust:
            lines.append(f"- `{sid}`")
        lines.append("")
    if borderline:
        lines.append("### ⚠ Borderline — passes at 0 bps, fails at 8 bps")
        lines.append("")
        lines.append("These strategies need follow-up: either improve entry quality, reduce trade count, or accept demotion.")
        for sid in borderline:
            lines.append(f"- `{sid}`")
        lines.append("")
    if broken:
        lines.append("### ❌ Fails at both tiers")
        lines.append("")
        lines.append("These were promoted under earlier (less strict) gates or by mistake. Candidates for retirement.")
        for sid in broken:
            lines.append(f"- `{sid}`")
        lines.append("")
    lines.append("---")
    lines.append("*Generated by `scripts/b5_walkforward_cost.py`.*")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default=None,
                     help="Comma-separated strategy IDs (default: all promoted)")
    ap.add_argument("--tiers", default="0,8",
                     help="Comma-separated bps tiers (default: 0,8)")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    tiers = [int(t) for t in args.tiers.split(",")]
    if args.strategies:
        ids = args.strategies.split(",")
    else:
        strategies, _ = validate_all()
        ids = [s.id for s in strategies if s.status in ("promoted", "live-ready", "live")]

    print(f"Running B5 on {len(ids)} strategies × {len(tiers)} tiers...", file=sys.stderr)
    results: list[dict] = []
    for sid in ids:
        # Need profile for gate check
        from strategy_engine.registry.loader import load_one
        from strategy_engine.backtest.runner import _find_yaml_path
        strat = load_one(_find_yaml_path(sid))
        profile = _guess_profile(strat)
        print(f"\n=== {sid}  [{profile}] ===", file=sys.stderr)
        for bps in tiers:
            wf, err = run_one(sid, bps)
            if err:
                print(f"  {bps:>3} bps: ERROR — {err}", file=sys.stderr)
                results.append({
                    "strategy_id": sid, "profile": profile, "bps": bps,
                    "error": err, "metrics": None, "passed": None,
                })
                continue
            passed, failed = _check_wf_gates(wf, profile)
            metrics = {
                "oos_active_mean_sharpe": wf.oos_active_mean_sharpe,
                "oos_active_bar_mean_sharpe": wf.oos_active_bar_mean_sharpe,
                "oos_active_mean_win_rate": wf.oos_active_mean_win_rate,
                "activation_rate": wf.activation_rate,
                "oos_total_trades": wf.oos_total_trades,
                "oos_worst_dd": wf.oos_worst_dd,
                "n_active_folds": wf.n_active_folds,
                "n_folds": wf.n_folds,
            }
            sharpe_key = "oos_active_bar_mean_sharpe" if profile == "active-trader" else "oos_active_mean_sharpe"
            verdict = "PASS" if passed else f"FAIL ({','.join(failed)})"
            print(f"  {bps:>3} bps: Sharpe={metrics[sharpe_key]:.3f}  active={wf.n_active_folds}/{wf.n_folds}  "
                  f"trades={wf.oos_total_trades}  wr={wf.oos_active_mean_win_rate:.1%}  "
                  f"dd={wf.oos_worst_dd:.2%}  {verdict}", file=sys.stderr)
            results.append({
                "strategy_id": sid, "profile": profile, "bps": bps,
                "metrics": metrics, "passed": passed, "failed_gates": failed,
            })

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        report = render(results)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report)
            print(f"\n✓ wrote report: {args.output}", file=sys.stderr)
        else:
            print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
